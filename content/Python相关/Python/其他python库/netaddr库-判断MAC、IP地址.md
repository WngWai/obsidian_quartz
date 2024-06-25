
## 实例
### 判断MAC
```python
from netaddr import EUI, mac_unix

def is_valid_mac(mac):
    try:
        eui = EUI(mac)
        eui.dialect = mac_unix
        return True
    except ValueError:
        return False

mac_address = "00:1A:2B:3C:4D:5E"
if is_valid_mac(mac_address):
    print("Valid MAC address")
else:
    print("Invalid MAC address")
```
### 判断IP
```python
from netaddr import IPAddress

def is_valid_ip(ip):
    try:
        IPAddress(ip)
        return True
    except ValueError:
        return False

ip_address = "192.168.0.1"
if is_valid_ip(ip_address):
    print("Valid IP address")
else:
    print("Invalid IP address")
```


## 常用函数
### netaddr.EUI()
在Python的netaddr库中，`netaddr.EUI()`函数用于创建和操作扩展唯一标识符（EUI），即MAC地址或其他类型的唯一标识符。
**函数定义**：
```python
netaddr.EUI(value=None, version=None, dialect=None, word_size=None, oui=None, **kwargs)
```
**参数**：
- `value`：可选参数，用于指定EUI的初始值。可以是字符串形式的MAC地址，也可以是整数或长整数。
- `version`：可选参数，用于指定EUI的版本。默认为None，表示根据提供的值自动确定版本。
- `dialect`：可选参数，用于指定EUI的方言。默认为None，表示使用默认方言。
- `word_size`：可选参数，用于指定EUI的字大小（Word Size）。默认为None，表示根据提供的值自动确定字大小。
- `oui`：可选参数，用于指定EUI的组织唯一标识符（Organizationally Unique Identifier，OUI）。
**示例**：
```python
import netaddr

# 示例：创建EUI对象
eui1 = netaddr.EUI('00-11-22-33-44-55')
eui2 = netaddr.EUI(1234567890, version=48)

# 打印EUI对象
print(eui1)
print(eui2)

# 示例：操作EUI对象
eui3 = netaddr.EUI('00-11-22-33-44-55')
eui4 = netaddr.EUI('66-77-88-99-AA-BB')

# 获取EUI的OUI
oui = eui3.oui

# 获取EUI的MAC地址
mac_address = eui4.dialect.format_eui(eui4)

# 打印OUI和MAC地址
print(oui)
print(mac_address)
```
在示例中，我们首先使用`netaddr.EUI()`函数创建了两个EUI对象，`eui1`和`eui2`。`eui1`使用字符串形式的MAC地址创建，`eui2`使用整数表示的值和指定的版本创建。然后，我们打印这些EUI对象的结果。
接下来，我们创建了两个EUI对象，`eui3`和`eui4`，用于演示EUI对象的操作。我们使用`eui3.oui`访问`eui3`的OUI（组织唯一标识符），并将其赋值给`oui`变量。我们还使用`eui4.dialect.format_eui(eui4)`获取`eui4`的MAC地址，并将其赋值给`mac_address`变量。最后，我们打印获得的OUI和MAC地址。


### netaddr.IPAddress()
在Python的netaddr库中，`netaddr.IPAddress()`函数用于创建和操作IP地址对象。
**函数定义**：
```python
netaddr.IPAddress(value=None, version=None, flags=0, **kwargs)
```
**参数**：
- `value`：可选参数，用于指定IP地址的初始值。可以是字符串形式的IP地址，也可以是整数或长整数。
- `version`：可选参数，用于指定IP地址的版本。默认为None，表示根据提供的值自动确定版本。
- `flags`：可选参数，用于指定标志位。默认为0。
**示例**：
```python
import netaddr

# 示例：创建IPAddress对象
ip1 = netaddr.IPAddress('192.168.0.1')
ip2 = netaddr.IPAddress(3232235521, version=4)

# 打印IPAddress对象
print(ip1)
print(ip2)

# 示例：操作IPAddress对象
ip3 = netaddr.IPAddress('192.168.0.1')
ip4 = netaddr.IPAddress('2001:db8::1')

# 获取IP地址的版本
version = ip3.version

# 检查IP地址是否为私有地址
is_private = ip4.is_private()

# 打印IP地址版本和私有地址检查结果
print(version)
print(is_private)
```

在示例中，我们首先使用`netaddr.IPAddress()`函数创建了两个IPAddress对象，`ip1`和`ip2`。`ip1`使用字符串形式的IPv4地址创建，`ip2`使用整数表示的值和指定的版本创建。然后，我们打印这些IPAddress对象的结果。

接下来，我们创建了两个IPAddress对象，`ip3`和`ip4`，用于演示IPAddress对象的操作。我们使用`ip3.version`获取`ip3`的IP地址版本，并将其赋值给`version`变量。我们还使用`ip4.is_private()`检查`ip4`是否为私有地址，并将结果赋值给`is_private`变量。最后，我们打印获得的IP地址版本和私有地址检查结果。

netaddr库还提供了其他功能，例如对IP地址进行转换、格式化、子网掩码操作、验证和计算网络地址等。您可以参考netaddr库的文档以了解更多详细信息和用法示例。