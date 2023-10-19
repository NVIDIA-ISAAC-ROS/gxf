'''
Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
'''

import numpy
import sys
import construct as cs

class Const:
    '''
    Helper class to store constants
    '''
    POSE_FRAME_NAME_SIZE = 64


class Cask():
    '''
    Helper class to parse cask files which were written by gxf::EntityRecorder

    NOTE: Currently only the gxf::Tensor component is supported. All other components are currently
          ignored.
    NOTE: All components which come after the first unsupported component are currently ignored.

    Requirements:
        pip install construct

    Example:
        cask = Cask(filename)
        print("Cask with {} entities".format(len(cask)))
        print(cask[0])
    '''

    def __init__(self, filename: str = None):
        if filename:
            self.load(filename)
        else:
            self.index = []
            self.entities_file = None
        # Initialize header structs
        # Component header struct
        self.fmt_component_header = cs.Struct(
            "serialized_size" / cs.Int64ul,
            "tid1" / cs.Int64ul,
            "tid2" / cs.Int64ul,
            "name_size" / cs.Int64ul)
        # Entity header struct
        self.fmt_entity_header = cs.Struct(
            "serialized_size" / cs.Int64ul,
            "checksum" / cs.Int32ul,
            "sequence_number" / cs.Int64ul,
            "flags" / cs.Int32ul,
            "component_count" / cs.Int64ul,
            "reserved" / cs.Int64ul)
        # Timestamp component header
        self.fmt_component_timestamp_header = cs.Struct(
            "pubtime" / cs.Int64sl,
            "acqtime" / cs.Int64sl)
        # Tensor component header
        self.fmt_component_tensor_header = cs.Struct(
            "storage_type" / cs.Int32sl,
            "element_type" / cs.Int32sl,
            "bytes_per_element" / cs.Int64ul,
            "rank" / cs.Int32sl,
            "dims" / cs.Array(8, cs.Int32sl),
            "strides" / cs.Array(8, cs.Int64ul))
        # Define list of numpy datatypes
        self.numpy_data_type = [
            numpy.dtype(numpy.uint8),   # kCustom
            numpy.dtype(numpy.int8),   # kInt8
            numpy.dtype(numpy.uint8),   # kUnsigned8
            numpy.dtype(numpy.int16),   # kInt16
            numpy.dtype(numpy.uint16),   # kUnsigned16
            numpy.dtype(numpy.int32),   # kInt32
            numpy.dtype(numpy.uint32),   # kUnsigned32
            numpy.dtype(numpy.int64),   # kInt64
            numpy.dtype(numpy.uint64),   # kUnsigned64
            numpy.dtype(numpy.float32),   # kFloat32
            numpy.dtype(numpy.float64)   # kFloat64
        ]


    def load(self, filename: str):
        '''
        Reads a GXF Cask file index written by gxf::FileStream

        The function loads the index file (*.gxf_index) which is simply a list where each element
        contains index data about an entity in the cask. The index contains three values:
        - timestamp: the timestamp of the entity
        - size: the length of the serialized entity in bytes
        - offset: the offset into the file where the serialized entity is stored (in bytes)

        The function also opens the entities file (*.gxf_entities) ready to be accessed.
        '''
        with open(filename + ".gxf_index", mode='rb') as index_file:
            self.index = numpy.fromfile(index_file, dtype=numpy.uint64).reshape((-1, 3))

        # Open file with entities
        self.entities_file = open(filename + ".gxf_entities", mode='rb')


    def __del__(self):
        if self.entities_file:
            self.entities_file.close()


    def _deserialize_component_header(self):
        # Setup the parsing struct for the component header needed by Construct
        component_header = self.fmt_component_header.parse_stream(self.entities_file)

        # read name
        name_bytes = self.entities_file.read(component_header.name_size)

        return {
          'tid': hex(component_header.tid1) + ' ' + hex(component_header.tid2),
          'name': name_bytes.decode("utf-8")
        }


    def _deserialize_entity(self, index: int):
        # Get the offset of the requested element (third value in the index)
        entity_file_offset = self.index[index][2]

        self.entities_file.seek(entity_file_offset)

        # Deserialize entity header
        entity_header = self.fmt_entity_header.parse_stream(self.entities_file)

        result = {}

        for i in range(entity_header.component_count):
            component_header = self._deserialize_component_header()

            # Deserialize components
            try:
                tensor = self._deserialize_component(component_header['tid'])
                result[component_header['name']] = tensor
            except:
                # only process known components and ignore the rest
                break

        return result


    def _deserialize_component(self, tid):
        if tid == '0x377501d69abf447c 0xa6170114d4f33ab8':   # TID of gxf::Tensor
            return self._deserialize_component_tensor()
        elif tid == '0xd1095b105c904bbc 0xbc89601134cb4e03':   # TID of gxf::Timestamp
            return self._deserialize_component_timestamp()
        else:
            raise Exception("Unsupported component type: {}".format(tid))


    def _deserialize_component_timestamp(self):
        # Parse the timestamp component header
        header = self.fmt_component_timestamp_header.parse_stream(self.entities_file)
        return {"pubtime": header.pubtime, "acqtime": header.acqtime}

    def _deserialize_component_tensor(self):
        # Parse the Tensor component header
        header = self.fmt_component_tensor_header.parse_stream(self.entities_file)

        # Ignore unnecessary 1's in the dimensions. E.g. [6,1,5,1,1,1] => [6,1,5]
        shape = header.dims
        shape.reverse()
        first_not_one = next((i for i, j in enumerate(shape) if j > 1), -1)
        shape = shape[first_not_one:]
        shape.reverse()

        count = numpy.prod(shape)

        if header.element_type == 0:
            raise Exception("Unsupported tensor primitive type: Custom")
        dtype = self.numpy_data_type[header.element_type]

        size = count * dtype.itemsize

        tensor = numpy.frombuffer(buffer=self.entities_file.read(size), dtype=dtype, count=count)
        return tensor.reshape(shape)


    def __len__(self):
        return len(self.index)


    def __getitem__(self, index: int):
        return self._deserialize_entity(index)


def test_cask(cask):
    print("Cask with {} entities".format(len(cask)))
    if len(cask) == 0:
        return
    print("The first element:")
    print(cask[0])

if __name__ == "__main__":
    cask = Cask(sys.argv[1])
    test_cask(cask)
