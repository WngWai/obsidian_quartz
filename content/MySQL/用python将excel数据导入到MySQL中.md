是的，你可以使用编程语言（如Python）和MySQL连接库（如mysql-connector-python）来实现将多个Excel表的数据逐步导入到数据库中的功能。下面是一个简单的示例代码，展示了如何使用Python和pandas库读取Excel表并将数据导入到MySQL数据库中：

```python
import pandas as pd
import mysql.connector

# 连接MySQL数据库
cnx = mysql.connector.connect(user='your_username', password='your_password',
                              host='your_host', database='your_database')

# 从Excel表中读取数据，并逐行导入到数据库
def import_excel_to_mysql(excel_file):
    # 读取Excel表，并指定需要导入的工作表
    df = pd.read_excel(excel_file, sheet_name='Sheet1')
    
    # 遍历每一行数据
    for row in df.itertuples():
        data = (row.column1, row.column2)  # 替换column1和column2为实际的列名
        
        # 执行插入数据的SQL语句
        cursor = cnx.cursor()
        insert_query = "INSERT INTO your_table (column1, column2) VALUES (%s, %s)"  # 替换your_table和column1、column2为实际的表名和列名
        cursor.execute(insert_query, data)
        
        cnx.commit()
        cursor.close()

# 调用函数导入多个Excel表
excel_files = ['excel_file1.xlsx', 'excel_file2.xlsx', 'excel_file3.xlsx']
for file in excel_files:
    import_excel_to_mysql(file)

# 关闭数据库连接
cnx.close()
```

在这个示例代码中，你需要将`your_username`、`your_password`、`your_host`、`your_database`替换为你的MySQL数据库的实际信息，以及将`column1`、`column2`、`your_table`替换为你需要导入数据的实际列名和表名。

请注意，此示例仅在导入单个工作表的数据时进行了演示。如果你的Excel文件中有多个工作表，你可能需要根据需要进行修改以适应不同的情况。



#### 示例
[Excel 批量导入Mysql(创建表-追加数据)_51CTO博客_mysql批量导入数据](https://blog.51cto.com/u_14195239/5608048)