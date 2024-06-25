# DML (Data Manipulation Language) 数据操作语言
对`表中数据`进行操作

**INSERT INTO、UPDATA**都直接跟表对象！！！**DELETE**不直接接表，用form接表。

## 基础语法
### 单一添加INSERT INTO...VALUES
```MySQL
INSERT INTO tb_name (field_name1,field_name2... ) VALUES (值1,值2...); -- 对指定字段添加数据
INSERT INTO tb_name VALUES (值1, 值2, ...);  -- 对全部字段添加数据
```

### 批量添加INSERT INTO...VALUES
```MySQL
-- 指定字段
INSERT INTO tb_name (字段名1, 字段名2, ...) 
VALUES (值1, 值2, ...), (值1, 值2, ...), (值1, 值2, ...); 

INSERT INTO 目标表 (列1, 列2, 列3, ...)
SELECT 列1, 列2, 列3, ...
FROM 源表
WHERE 条件;

-- 全部字段
INSERT INTO tb_name 
VALUES (值1, 值2, ...), (值1, 值2, ...), (值1, 值2, ...); 

-- 从源表中插入内容
INSERT INTO 目标表 
SELECT *
FROM 源表
WHERE 条件;

INSERT INTO 目标表 (列1, 列2, 列3)  
SELECT 列1, 列2, 列 as 列3  
FROM 源表  
WHERE 条件;
```

要确保源表和目标表的结构**兼容**，也就是说，**源表的每一列都要与目标表的对应列有相同的数据类型和长度**。否则，复制操作可能会失败。
列名要求相同！？

### 改UPDATE...SET
UPDATE 相当于WHERE的作用，所以后面可以接表连接。
```MySQL
-- 单表更新
UPDATE tb_name 
SET 字段名1 = 值1, 字段名2 = 值2, ... 
[ WHERE id = 1];  

-- 多表连接。虽然进行了表连接，但看set部分的内容，是更行了别名为o表的数据！
UPDATE orders o JOIN customers c ON o.customer_id = c.id 
SET o.customer_name = c.name;

-- 子查询
UPDATE table_a a SET a.column_name = ( SELECT b.column_name FROM table_b b WHERE a.matching_column = b.matching_column )
```

建议多用**子查询**向目标表中补充内容，直观明白！

注意事项

- 字符串和日期类型数据应该包含在**引号中**

- 插入的数据大小应该在字段的规定**范围内**

- 多内容更新，在SET语句中应该用逗号，而非and！

### 删DELETE
根据筛选条件，删除指定表中符合要求的行数据，**只能整行删除**呀！
**没有条件**，将表中数据全部删除。

```MySQL
DELETE 
FROM 表名 
[ WHERE 条件 ]; 
```

## 常用操作
### [[插入时的主键重复问题]]

### 借助子查询
删除
```MySQL
DELETE 
FROM students 
WHERE id IN (SELECT id FROM students WHERE grade = 'not_selected');
```