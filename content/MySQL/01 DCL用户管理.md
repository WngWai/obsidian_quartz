
# CL (Data Control Language) 数据控制语言

## 管理用户
```mysql
-- 查询用户：
USER mysql;
SELECT * FROM user;
-- 创建用户:  
CREATE USER '用户名'@'主机名' IDENTIFIED BY '密码';
-- 修改用户密码：  
ALTER USER '用户名'@'主机名' IDENTIFIED WITH mysql_native_password BY '新密码';
-- 删除用户：  
DROP USER '用户名'@'主机名';
```


## 权限控制

```MySQL
-- 查询权限：  
SHOW GRANTS FOR '用户名'@'主机名';
-- 授予权限：  
GRANT 权限列表 ON 数据库名.表名 TO '用户名'@'主机名';
-- 撤销权限：  
REVOKE 权限列表 ON 数据库名.表名 FROM '用户名'@'主机名';
```

注意事项

- 主机名可以使用 % 通配

- 多个权限用逗号分隔

- 授权时，数据库名和表名可以用 * 进行通配，代表所有

| 权限 | 说明 |
| ---- | ---- |
| ALL, ALL PRIVILEGES | 所有权限 |
| SELECT | 查询数据 |
| INSERT | 插入数据 |
| UPDATE | 修改数据 |
| DELETE | 删除数据 |
| ALTER | 修改表 |
| DROP | 删除数据库/表/视图 |
| CREATE | 创建数据库/表 |
