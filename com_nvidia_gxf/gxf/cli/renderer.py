# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from abc import ABC, abstractmethod
import copy
from typing import Callable, List
from result import Result, Ok, Err

def truncate_string(s, length):
    if len(s) <= length:
        return s
    else:
        n = (length - 3) // 2
        return s[:n] + "..." + s[-n:]
class DataSource:
    """
    Base class of data source for fetching data
    Args:
        name     :   name of the data source
        desc     :   description of the data source
        update_fn:   Callable to fetch data
        filterables: list of strings that can be used as keys for filtering data
    """
    def __init__(self, name: str, desc: str, update_fn: Callable, filterable: List):
        self._name = name
        self._description = desc
        self._update_fn = update_fn
        self._data = None
        self._filter = None
        self._filter_list = filterable

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return self._description

    @property
    def data(self):
        return self._data

    def update(self):
        self._data = self._update_fn(self._filter)

    def create_table(self):
        pass

    def filter_list(self):
        return copy.copy(self._filter_list)

    def set_filter(self, filter: str) -> Result:
        if not filter:
            self._filter = None
            return Ok('')
        pair = filter.split('=')
        if len(pair) != 2 or not pair[1]:
            return Err("Wrong Format!")
        if not pair[0] in self._filter_list:
            return Err("Wrong Field")
        self._filter = pair
        return Ok('')

    def get_filter(self):
        return self._filter

class StringTableDataSource(DataSource):
    """
    The data source wraps a list of string list which can be displayed in a
    2-d table
    Args:
        name     :   name of the data source
        desc     :   description of the data source
        header   :   header format for being showed as a table
        update_fn:   Callable to fetch data
        filterables: list of strings that can be used as keys for filtering data
    """
    def __init__(self, name, desc, header, update_fn, filterable):
        super().__init__(name, desc, update_fn, filterable)
        self._header = header
        self._data = []

    @property
    def header(self):
        return self._header

    def create_table(self):
        return StringTablePanel(self)

class CompoundStringTableDataSource(DataSource):
    """
    The data source multi-level string table data
    Args:
        name     :   name of the data source
        desc     :   description of the data source
        header   :   header format for being showed as a table
        update_fn:   Callable to fetch data
        filterables: list of strings that can be used as keys for filtering data
    """
    def __init__(self, name, desc, header, update_fn, filterable):
        super().__init__(name, desc, update_fn, filterable)
        self._header = header
        self._data = []

    @property
    def header(self):
        return self._header

    def create_table(self):
        return CompoundStringTablePanel(self)

class Renderer(ABC):
    """
    Base class for renderer
    """
    def __init__(self, data: DataSource, refresh_ms):
        self._data_source = data
        self._refresh_ms = refresh_ms

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

import urwid

class StringTablePanel(urwid.WidgetWrap):
    """
    The class wraps a urwid widget to display 2-d string table
    Args:
        data: data source of the string table
    """
    def __init__(self, data: StringTableDataSource):
        data.update()
        self.data_source = data
        urwid.WidgetWrap.__init__(self, self._build_ui())

    def refresh(self):
        # update the data
        self.data_source.update()
        self.content.clear()
        for l in self.data_source.data:
            text = ""
            for i in l:
                maxlen = i[1]
                label = truncate_string(str(i[0]), int(maxlen)-1)
                text += "%-*s" % (maxlen, label)
            self.content.append(urwid.Text(('table_content', text)))

    def _build_ui(self) -> urwid.WidgetWrap:
        """create the UI layout"""
        header_text = ""
        for h in self.data_source.header:
            header_text += "%-*s" % (h[2], h[1])
        title_text = '{:^}'.format(self.data_source.description)
        header = urwid.Pile([
            urwid.Text(('title', title_text)),
            urwid.Text(('table_header', header_text))
        ])
        self.content = urwid.SimpleFocusListWalker([])

        table_view = urwid.ListBox(self.content)
        frame = urwid.Frame(table_view, header=header)
        return urwid.LineBox(frame)

class CompoundStringTablePanel(urwid.WidgetWrap):
    """
    The class wraps multiple string table into one group
    """
    def __init__(self, data: CompoundStringTableDataSource):
        data.update()
        self.data_source = data
        self.content = dict()
        urwid.WidgetWrap.__init__(self, self._build_ui())

    def refresh(self):
        self.data_source.update()
        # clear all the contents
        for k in self.content:
            for j in self.content[k]:
                self.content[k][j].clear()

        # reconnected. need to rebuild the entire UI
        if not self.content and self.data_source.data:
            self._set_w(self._build_ui())

        for key, data in self.data_source.data.items():
            for sub_key, sub_data in data.items():
                content = self.content[key][sub_key]
                for l in sub_data:
                    text = ""
                    for i in l:
                        maxlen = i[1]
                        label = truncate_string(str(i[0]), int(maxlen)-1)
                        text += "%-*s" % (maxlen, label)
                    content.append(urwid.Text(('table_content', text)))

    def _build_ui(self) -> urwid.WidgetWrap:
        title_text = '{:^}'.format(self.data_source.description)
        title = urwid.Text(('title', title_text))
        table_header_text = ""
        for h in self.data_source.header:
            table_header_text += "%-*s" % (h[2], h[1])
        rows = []
        for key, data in self.data_source.data.items():
            top_header =urwid.Text(('title', key), align=urwid.CENTER)
            widgets = []
            sub_content = dict()
            for sub_key in data:
                table_header = urwid.Pile([
                    urwid.Text(('table_header', sub_key)),
                    urwid.Text(('table_header', table_header_text))
                ])
                content = urwid.SimpleFocusListWalker([])
                table_view = urwid.ListBox(content)
                sub_content[sub_key] = content
                widgets.append(
                    urwid.Frame(table_view, header=table_header)
                )
            self.content[key] = sub_content
            frame = urwid.Frame(urwid.Columns(widgets), header=top_header)
            rows.append(urwid.LineBox(frame))
        return urwid.Frame(urwid.Pile(rows), header=title)

class FilterInput(urwid.Filler):
    def __init__(self, renderer, body, valign):
        super().__init__(body, valign)
        self._renderer = renderer
    def keypress(self, size, key):
        if key != 'enter':
            return super(FilterInput, self).keypress(size, key)
        else:
            result = self._renderer.apply_filter(self.body.edit_text)
            if (result.is_err()):
                self.body.edit_text = result.value

class UrwidRenderer(Renderer):
    """
    Text based renderer on console output using urwid
    Args:
        data: list of data sources for rendering
        refresh_ms: refresh period in milliseconds
    """
    palette = [
        ('title', 'dark green,bold', ''),
        ('table_content', 'white', ''),
        ('table_header', 'yellow,bold', ''),
        ('normal', 'white', ''),
        ('foot', 'black', 'dark cyan'),
    ]

    def __init__(self, data: DataSource, refresh_ms=1000):
        super().__init__(data, refresh_ms)
        self._build_ui()
        self._main_loop = urwid.MainLoop(
            self._main_widget, self.palette, unhandled_input=self.handle_input)
        self._refresh(self._main_loop, None)

    def _build_ui(self):
        self._table_widget = self._data_source.create_table()
        f1 = urwid.Button([('normal', u'F1'), ('foot', u'Help')])
        urwid.connect_signal(f1, 'click', lambda k:self.on_help())
        f2 = urwid.Button([('normal', u'F2'), ('foot', u'Filter')])
        urwid.connect_signal(f2, 'click', lambda k:self.on_filter_input())
        self._buttons = urwid.Columns([f1, f2])
        self._main_widget = urwid.Frame(self._table_widget, footer=self._buttons)

    def _refresh(self, loop, data):
        # refresh the table
        self._table_widget.refresh()
        self._main_loop.set_alarm_in(self._refresh_ms/1000, self._refresh)

    def handle_input(self, key):
        if type(key) == str:
            if key in ('q', 'Q'):   
                raise urwid.ExitMainLoop()
            if key == 'f1':
                self.on_help()
            elif key == 'f2':
                self.on_filter_input()
            else:
              self._main_loop.widget = self._main_widget
        elif type(key) == tuple:
            pass

    def on_help(self):
        from gxf.cli.gxf_cli import VERSION
        help_txt = \
            f"""
        gxf_cli {VERSION} 

        """
        help_widget = urwid.Text([('normal', help_txt),
                                    ('foot', u'\nPress any key to return')],
                                    align='left')
        fill = urwid.Filler(help_widget, 'top')
        self._main_loop.widget = fill

    def on_filter_input(self):
        fields = ""
        filterable = self._data_source.filter_list()
        if filterable:
            for filter in filterable:
                fields += f"{filter},"
            caption = f"Set a filter, supporting [{fields}]:"
            filter = self._data_source.get_filter()
            filter_text = f"{filter[0]}={filter[1]}" if filter else f"{filterable[0]}="
            fill = FilterInput(self, urwid.Edit(caption=caption, edit_text=filter_text), 'top')
            self._main_loop.widget = fill

    def apply_filter(self, filter:str) -> Result:
        res = self._data_source.set_filter(filter)
        if res.is_err():
            return res

        self._build_ui()
        self._main_loop.widget = self._main_widget
        return Ok('')

    def start(self):
        self._main_loop.run()

    def stop(self):
        raise urwid.ExitMainLoop()
