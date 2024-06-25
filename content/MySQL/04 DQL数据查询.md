# DQL (Data Query Language) 数据查询语言
## 基础语法

[[语句执行顺序]]
```MySQL
-- 执行顺序：
FROM -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY -> LIMIT

FROM,WHERE,
GROUP BY (聚合函数), HAVING,
SELECT,DISTINCT,
ORDER BY,LIMIT
```


```MySQL
-- 编写顺序：
select 字段列表 
from 表名列表 
where 条件列表 
group by 分组字段列表 
having 分组后条件列表 
order by 排序字段列表
limit 分页参数；

-- 1，基础查询
select field_name1,field_name2... from tb_name;  -- 查询多个字段
select * from tb_name; -- 查询所有字段

select field_name1 [as filed_name2] from tb_name; -- 设置别名1
select field_name1 filed_name2 from tb_name; -- 设置别名2

select distinct field_name from tb_name; -- 去重

-- 2，条件查询
 select field_name from tb_name where 条件列表； -- 细看下面的举例
 
-- 聚合函数
SELECT 聚合函数(field_name1,field_name2...) FROM tb_name; -- NULL值不参与聚合函数

-- 分组查询
SELECT field_name1,field_name2... FROM tb_name [ WHERE 条件列表 ] GROUP BY 分组字段名 [ HAVING 分组后的过滤条件 ]; -- where筛选，再分组，执行聚合函数，再根据having二次筛选

-- 排序查询
SELECT field_name1,field_name2... FROM tb_name ORDER BY 字段1 排序方式1, 字段2 排序方式2; -- ASC: 升序（默认）;DESC: 降序

-- 分页查询
语法：  
SELECT field_name1,field_name2... FROM tb_name LIMIT 起始索引, 查询记录数;
```

[[LIMIT]]
### 1，基础查询

`都不行！`MySQL中**没有直接的语句**可以选择表中除了某些列之外的所有列。只能一个个输入！
```MySQL
-- 筛选部分列
SELECT * EXCEPT (column_to_exclude)
FROM your_table;

-- 筛选部分列
SELECT * -column_to_exclude
FROM your_table;
```

[[CAST]]数据类型转换


### 2，条件查询
转义：
`SELECT * FROM 表名 WHERE name LIKE '/_张三' ESCAPE '/'`  

/ 之后的\_不作为通配符  

##### 比较运算符
| 比较运算符          | 功能                                        |     |
| ------------------- | ------------------------------------------- | --- |
| >                   | 大于                                        |     |
| >=                  | 大于等于                                    |     |
| <                   | 小于                                        |     |
| <=                  | 小于等于                                    |     |
| =                   | 等于                                        |     |
| <> 或 !=            | 不等于                                      |     |
| BETWEEN ... AND ... | 在某个范围内（含最小、最大值）              |     |
| IN(...)             | 在in之后的列表中的值，多选一                |     |
| LIKE 占位符         | 模糊匹配（\_匹配单个字符，%匹配任意个字符） |     |
| IS NULL             | 是NULL                                      |     |

IN 和 NOT IN
IS NULL 和IS NOT NULL


where `交易卡号`IN ('621582xxxxxxxxxxxxx', '6228480xxxxxxxxxxxxx')

#### 逻辑运算符
| 逻辑运算符         | 功能                         |
| ------------------ | ---------------------------- |
| AND 或 &&          | 并且（多个条件同时成立）     |
| OR 或 &#124;&#124; | 或者（多个条件任意一个成立） |
| NOT 或 !           | 非，不是                     |

例子：

```mysql
-- 年龄等于30
select * from employee where age = 30;
-- 年龄小于30
select * from employee where age < 30;
-- 小于等于
select * from employee where age <= 30;
-- 没有身份证
select * from employee where idcard is null or idcard = '';
-- 有身份证
select * from employee where idcard;
select * from employee where idcard is not null;
-- 不等于
select * from employee where age != 30;
-- 年龄在20到30之间
select * from employee where age between 20 and 30;
select * from employee where age >= 20 and age <= 30;
-- 下面语句不报错，但查不到任何信息
select * from employee where age between 30 and 20;
-- 性别为女且年龄小于30
select * from employee where age < 30 and gender = '女';
-- 年龄等于25或30或35
select * from employee where age = 25 or age = 30 or age = 35;
select * from employee where age in (25, 30, 35);
-- 姓名为两个字
select * from employee where name like '__';
-- 身份证最后为X
select * from employee where idcard like '%X';
select * from employee where idcard like '_________________X';
```

#### 算术运算符
\+ \- \* \/


### 3，分组查询GROUP BY
where 和 having 的区别：

- 执行时机不同：where是**分组之前进行过滤**，不满足where条件**不参与分组**；having是**分组后对结果进行过滤**。

- 判断条件不同：where**不能对聚合函数**进行判断，而having可以。

 注意事项：
 
- 执行顺序：**where > 聚合函数 > having**

- 分组之后，查询的字段一般为聚合函数和分组字段，查询其他字段无任何意义

例子：

```mysql
-- 根据性别分组，统计男性和女性数量（只显示分组数量，不显示哪个是男哪个是女）
select count(*) from employee group by gender;
-- 根据性别分组，统计男性和女性数量
select gender, count(*) from employee group by gender;
-- 根据性别分组，统计男性和女性的平均年龄
select gender, avg(age) from employee group by gender;
-- 年龄小于45，并根据工作地址分组
select workaddress, count(*) from employee where age < 45 group by workaddress;
-- 年龄小于45，并根据工作地址分组，获取员工数量大于等于3的工作地址
select workaddress, count(*) address_count from employee where age < 45 group by workaddress having address_count >= 3;
```

### 4，聚合查询（聚合函数）

常见聚合函数：

| 函数  | 功能     |
| ----- | -------- |
| count | 统计数量 |
| max   | 最大值   |
| min   | 最小值   |
| avg   | 平均值,不是“mean”   |
| sum   | 求和     |

**有时去重**用大DISTIN、MAX、MIN

例子：

```MySQL
-- count()
SELECT count(id) from employee where workaddress = "广东省";
SELECT count(*) from employee where workaddress = "广东省";
```


### 窗口函数（WINDOWS）
[[窗口函数]]，也叫OLAP函数（Online Anallytical Processing，联机分析处理）

通俗易懂的学会：SQL窗口函数 - 猴子的文章 - 知乎
https://zhuanlan.zhihu.com/p/92654574

**在每组内排名**，如下面的业务需求：

> 排名问题：每个部门按业绩来排名  
> topN问题：找出每个部门排名前N的员工进行奖励


```MySQL

<窗口函数> over (partition by <用于分组的列名>
                order by <用于排序的列名>)

```

<窗口函数>的位置，可以放以下两种函数：

1） 专用窗口函数，包括后面要讲到的rank, dense_rank, row_number等专用窗口函数。

2） 聚合函数，如sum. avg, count, max, min等

因为窗口函数是对where或者group by子句**处理后的结果**进行操作，所以窗口函数原则上只能写在select子句中。

### 5，排序查询ORDER BY
注意事项

如果是多字段排序，先对第一个字段进行排序，在次基础上，在相同的第一字段下再对第二字段进行排序；

例子：
```mysql
-- 根据年龄升序排序
SELECT * FROM employee ORDER BY age ASC;
SELECT * FROM employee ORDER BY age;
-- 两字段排序，根据年龄升序排序，入职时间降序排序(如果年龄相同那么就按这个)
SELECT * FROM employee ORDER BY age ASC, entrydate DESC;
```

### 6，分页查询LIMIT
例子：
```mysql
-- 查询第一页数据，展示10条
SELECT * FROM employee LIMIT 0, 10;
SELECT * FROM employee LIMIT    10;
-- 查询第二页
SELECT * FROM employee LIMIT 10, 10;
```

注意事项
- 起始索引从0开始，起始索引 = （**查询页码 - 1**） * 每页显示记录数

- 分页查询是**数据库的方言**，不同数据库**有不同实现**，MySQL是LIMIT

- 如果查询的是第一页数据，起始索引可以省略，直接简写 LIMIT 10，省略了0
