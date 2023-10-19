# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cc_library(
    name = "breakpad",
    srcs = [
        "src/client/linux/crash_generation/crash_generation_client.cc",
        "src/client/linux/dump_writer_common/thread_info.cc",
        "src/client/linux/dump_writer_common/ucontext_reader.cc",
        "src/client/linux/handler/exception_handler.cc",
        "src/client/linux/handler/minidump_descriptor.cc",
        "src/client/linux/log/log.cc",
        "src/client/linux/microdump_writer/microdump_writer.cc",
        "src/client/linux/minidump_writer/linux_dumper.cc",
        "src/client/linux/minidump_writer/linux_ptrace_dumper.cc",
        "src/client/linux/minidump_writer/minidump_writer.cc",
        "src/client/minidump_file_writer.cc",
        "src/common/convert_UTF.cc",
        "src/common/convert_UTF.h",
        "src/common/linux/elfutils.cc",
        "src/common/linux/file_id.cc",
        "src/common/linux/guid_creator.cc",
        "src/common/linux/linux_libc_support.cc",
        "src/common/linux/memory_mapped_file.cc",
        "src/common/linux/safe_readlink.cc",
        "src/common/md5.cc",
        "src/common/simple_string_dictionary.cc",
        "src/common/string_conversion.cc",
    ],
    hdrs = glob(["src/**/*.h"]),
    copts = ["-Wno-maybe-uninitialized"],
    strip_include_prefix = "src",
    visibility = ["//visibility:public"],
    deps = ["@lss"],
)

cc_library(
    name = "dump_syms-lib",
    srcs = glob([
        "src/common/dwarf/bytereader.cc",
        "src/common/dwarf/dwarf2diehandler.cc",
        "src/common/dwarf/dwarf2reader.cc",
        "src/common/dwarf/elf_reader.cc",
        "src/common/dwarf/elf_reader.h",
        "src/common/dwarf_cfi_to_module.cc",
        "src/common/dwarf_cu_to_module.cc",
        "src/common/dwarf_line_to_module.cc",
        "src/common/dwarf_range_list_handler.cc",
        "src/common/language.cc",
        "src/common/module.cc",
        "src/common/path_helper.cc",
        "src/common/stabs_reader.cc",
        "src/common/stabs_to_module.cc",
        "src/common/linux/crc32.cc",
        "src/common/linux/dump_symbols.cc",
        "src/common/linux/elf_symbols_to_module.cc",
        "src/common/linux/elfutils.cc",
        "src/common/linux/file_id.cc",
        "src/common/linux/linux_libc_support.cc",
        "src/common/linux/memory_mapped_file.cc",
        "src/common/linux/safe_readlink.cc",
        "src/tools/linux/dump_syms/dump_syms.cc",
    ]),
    hdrs = glob(["src/**/*.h"]),
    copts = [
        #https://en.wikipedia.org/wiki/Stabs
        #https://github.com/google/breakpad/blob/master/src/common/stabs_reader.cc
        #https://github.com/torvalds/linux/blob/master/include/uapi/linux/a.out.h#L159
        "-DN_UNDF=0x0",
        "-Wno-switch",
    ],
    strip_include_prefix = "src",
    deps = ["@lss"],
)

cc_binary(
    name = "dump_syms",
    visibility = ["//visibility:public"],
    deps = [":dump_syms-lib"],
)

cc_library(
    name = "libdisasm-lib",
    srcs = glob(["src/third_party/libdisasm/*.c"]),
    hdrs = glob(["src/third_party/libdisasm/*.h"]),
    copts = [
        "-Wno-format-truncation",
        "-Wno-stringop-truncation",
    ],
    strip_include_prefix = "src/third_party/libdisasm",
)

cc_library(
    name = "minidump_stackwalk-lib",
    srcs = [
        "src/processor/basic_code_modules.cc",
        "src/processor/basic_source_line_resolver.cc",
        "src/processor/call_stack.cc",
        "src/processor/cfi_frame_info.cc",
        "src/processor/convert_old_arm64_context.cc",
        "src/processor/disassembler_x86.cc",
        "src/processor/dump_context.cc",
        "src/processor/dump_object.cc",
        "src/processor/exploitability.cc",
        "src/processor/exploitability_linux.cc",
        "src/processor/exploitability_win.cc",
        "src/processor/logging.cc",
        "src/processor/minidump.cc",
        "src/processor/minidump_processor.cc",
        "src/processor/pathname_stripper.cc",
        "src/processor/proc_maps_linux.cc",
        "src/processor/process_state.cc",
        "src/processor/simple_symbol_supplier.cc",
        "src/processor/source_line_resolver_base.cc",
        "src/processor/stack_frame_cpu.cc",
        "src/processor/stack_frame_symbolizer.cc",
        "src/processor/stackwalk_common.cc",
        "src/processor/stackwalker.cc",
        "src/processor/stackwalker_address_list.cc",
        "src/processor/stackwalker_amd64.cc",
        "src/processor/stackwalker_arm.cc",
        "src/processor/stackwalker_arm64.cc",
        "src/processor/stackwalker_mips.cc",
        "src/processor/stackwalker_ppc.cc",
        "src/processor/stackwalker_ppc64.cc",
        "src/processor/stackwalker_sparc.cc",
        "src/processor/stackwalker_x86.cc",
        "src/processor/symbolic_constants_win.cc",
        "src/processor/tokenize.cc",
    ],
    hdrs = glob([
        "src/processor/*.h",
        "src/google_breakpad/processor/*.h",
        "src/google_breakpad/common/*.h",
        "src/third_party/libdisasm/libdis.h",
    ]),
    strip_include_prefix = "src",
    deps = [
        ":dump_syms-lib",
        ":libdisasm-lib",
    ],
)

cc_binary(
    name = "minidump_stackwalk",
    srcs = [
        "src/processor/minidump_stackwalk.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":minidump_stackwalk-lib"],
)
