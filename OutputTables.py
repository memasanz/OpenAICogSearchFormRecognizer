import json


        
class TableCell:  
    def __init__(self, index_row: int, index_column: int, text: str, row_span: int, column_span: int):  
        self.index_column = index_column  
        self.index_row = index_row  
        self.text = text   
        self.row_span = row_span
        self.column_span = column_span
        
class OutputTable:  
    def __init__(self, page_number: int, row_count: int, column_count: int):  
        self.page_number = page_number  
        self.row_count = row_count  
        self.column_count = column_count  
        self.cells = []  
  
    def add_record(self, data: TableCell):  
            self.cells.append(data) 
  
    def to_json(self):  
        table_dict = {  
            'page_number': self.page_number,  
            'row_count': self.row_count,  
            'column_count': self.column_count,  
            'cells': []  
        }  
  
        # convert table cells to dictionary  
        for cell in self.cells:  
            cell_dict = {  
                'index_row': cell.index_row, 
                'index_column': cell.index_column,   
                'text': cell.text, 
                'row_span': cell.row_span,
                'column_span': cell.column_span
            }  
            table_dict['cells'].append(cell_dict)  
  
        # convert dictionary to JSON  
        return json.dumps(table_dict)  
    
    def to_markdown(self):
        table_data = json.loads(self.to_json())
        #print(table_data)
        # create an empty table  
        table = []  
        for row in range(table_data['row_count']):  
            table.append([''] * table_data['column_count']) 
        


        # fill in table with cell data  
        for cell in table_data['cells']:  
            row_index = cell['index_row']  
            col_index = cell['index_column']  
            row_span = cell['row_span']  
            col_span = cell['column_span']  
            cell_text = cell['text']  

#           # fill in cell value  
            table[row_index][col_index] = cell_text  

             # span cell across rows  
            if row_span > 1:  
                for i in range(row_index + 1, row_index + row_span):  
                    table[i][col_index] = ''  

             # span cell across columns  
            if col_span > 1:  
                for j in range(col_index + 1, col_index + col_span):  
                    table[row_index][j] = ''  

        mark_down = ''
        for row in table:  
            row_text = '|'  
            for cell in row:  
                row_text += f' {cell} |'  
            mark_down = mark_down + (row_text) + '\n'
        return mark_down


class OutputTables:
    def __init__(self):
        self.tables = []
            
    def add_table(self, data: OutputTable):
        self.tables.append(data)