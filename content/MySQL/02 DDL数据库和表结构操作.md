# 1. DDL (Data Definition Language) 数据定义语言

CREATE TABLE，ALTER TABLE，DROP TABLE

## 基础语法
### **sql数据库**操作

```MySQL
-- 查所有数据库
show databases; 

-- 当前在哪个数据库下操作
select database(); 

-- 使用数据库，转换到需要操作的数据库对象
use database_name; 

-- （重要）创建数据库
create databases [if not exists] database_name charset utf8mb4; 

-- 删除数据库
drop databases [if exists] database_name; 
```


 ### **表结构**操作
#### 表操作
desc有描述的意思！

```MySQL
-- 查询当前数据库所有表
show tables; 

-- 查询表框架
desc tb_name;  

-- 看创建表的具体语句
show create table tb_name; 
```


#### CREATE TABLE创建表
```MySQL
create table tb_name(
id int primary key AUTO_INCREMENT [comment 字段注释],
name char(10) [comment 姓名],
age int [comment 年龄],
gender char(10) [comment 性别]
) comment 用户表; -- 最后一个字段后面没有逗号

# 通过select
CREATE TABLE new_table
SELECT column1, column2, ...
FROM old_table 
GROUP BY column1, column2, ...;
```

primary key中带有not null的性质，可以不用加not null。

- [[MySQL/base/字段（数据）类型|字段（数据）类型]]

- 常用约束：not null **非空值**约束；default默认约束；primary key**主键**约束

AUTO_INCREMENT 插入新内容时，主键数据**自动递增**

#### ALTER TABLE表内容的增改删
```MySQL
-- 修改表名
alter table tb_name rename to tb_name_new; 

-- 修改字段名和字段类型
alter table tb_name change id new_id char(10) [comment '用户名']; 

-- 修改数据类型
alter table tb_name modify id char(10); 

-- 添加字段
alter table tb_name add nickname char(10) [comment '昵称']; 

-- 删除字段
alter table tb_name drop id_change; 
```

change：必须带新列名，**新列的类型必须带**。

注意：只能在现有表上进行结构的操作，而不是在表的查询结果上进行结构操作


#### DROP TABLE删除表
```MySQL
-- 删除表
drop table [if exists] tb_name_new; 

-- 删除原表并创建一样表头的新表 
truncate table tb_name_new; 
```

truncate \[ˈtrʌŋˌkeɪt\] 清空表里的所有记录

# 经常用到的操作

## 根据旧表**生成一张新的表结构**
```MySQL
CREATE TABLE new_table AS SELECT * FROM old_table WHERE 1=0;
# AS可以省略
CREATE TABLE new_table SELECT * FROM old_table WHERE 1=0;

# 创建一个与现有表结构相同的空表
CREATE TABLE new_table LIKE old_table;
```

在此处的AS**用于创建新表**！AS可以省略

完整过程：使用相关sql文件，删除相应表，新建表
```MySQL
use `spider`;

drop table if exists 'tb top_movie';

create `tabletb top movie`(
	`mov id` int unsigned auto increment comment '编号',
	`title` varchar(50) not null comment '标题',
	`rating` decimal(3,1) not null comment'评分' ,
	`subject` varchar(200) default '主题' comment
	primary key(mov_id`)
)engine=innodb comment='Top电影表';
```


## 对排序后的表插入id
```MySQL

CREATE TABLE new_table LIKE old_table;

INSERT INTO new_table
SELECT *
FROM old_table
ORDER BY column1, column2, ...;

# 插入id后反而乱序了！
ALTER TABLE new_table ADD id INT PRIMARY KEY AUTO_INCREMENT;

# 得到排序后的id内容
SELECT 
    ROW_NUMBER() OVER (ORDER BY column1, column2, ...) AS id,  -- 使用ROW_NUMBER()生成ID
    a.*
FROM (
    SELECT *
    FROM 
        new_table
    ORDER BY 
       column1, column2, ...
) AS a;

```
不管是手动还是程序插入id结果都乱序了，说是存储的结构问题？？
最后还是依靠row_number()得到插入id后的查询，导出、再导入。