---
title: "LAMP 与阿里云服务器详解"
tags:
  - 云服务器
  - Web 部署
  - LAMP
categories: 云计算
lang: zh-CN
disableNunjucks: true
---

刚买的一台阿里云 ECS，从「能 SSH 登录」到「公网能稳定访问、跑得动一个站点」之间，最容易卡的其实就三件事：

1.  **网络通不通**——包可能在云厂商的安全组、操作系统防火墙、监听端口三个地方被悄悄丢掉，你看到的现象只有一个：浏览器一直转圈。
2.  **服务串不起来**——Apache、PHP、MySQL 是三个独立的进程，靠文件后缀、Unix socket、TCP 端口互相找到对方，每个接口都有自己的坑。
3.  **身份和权限不匹配**——Apache 跑在 `www-data` 用户下，MySQL 跑在 `mysql` 用户下，`wget` 下来的文件却归 `root` 所有。组合错了就是 403、Access denied，最后被逼到 `chmod 777`。

这篇文章就按你第一天会撞到的顺序把上面三件事讲透，再继续把第三十天才会遇到的问题——HTTPS、虚拟主机、备份、源码编译、什么时候该把单机拆成多机——一起讲完。目标是你照着做能跑起来，并且过半年回头看不会觉得自己当时埋了一堆雷。

## 读完这篇你能做到

-   在脑子里画出一个 HTTP 请求穿过 Linux、Apache、PHP、MySQL 的完整路径，知道每一跳可能在哪儿挂掉。
-   配置阿里云网络时不再用 `0.0.0.0/0` 一把梭，而是有真正的纵深防御。
-   在 Ubuntu 上装好、验通、加固 LAMP 的每一个组件（CentOS / Alibaba Cloud Linux 的差异点也会顺带说清）。
-   把一个真实的应用（Discuz!）从下载到上线全跑一遍，包括官方文档跳过的权限和数据库账号细节。
-   排查掉 90% 的 LAMP 故障所对应的那 5 个最常见错误。
-   判断什么时候应该一直留在单机，什么时候该拆成 SLB + ECS + RDS。

## 前提

-   一台阿里云 ECS，操作系统 Ubuntu 22.04 LTS 或 Alibaba Cloud Linux 3 / CentOS 7+。
-   能用密钥（不是密码）SSH 进去。
-   常用的 Linux 命令熟练：`ls`、`cd`、`cat`、`systemctl`、`sudo`。
-   一个域名是可选的，但讲到 HTTPS 那段会方便很多。

---

# 1. 为什么 2025 年还要学 LAMP

LAMP（**L**inux + **A**pache + **M**ySQL + **P**HP）从 2010 年开始就被各种新框架轮番宣告过死亡，但它每次都活了下来。原因不是怀旧，是「合适」。对内容站点、CMS（WordPress、Discuz、Drupal、MediaWiki）、客户门户、内部工具，以及一长串小型 SaaS 后端来说，LAMP 仍然是把动态网页放到用户面前**性价比最高、文档最全、维护成本最低**的方案。

LAMP 自带、新栈要自己拼起来的东西：

-   **极成熟的生态。** Apache 二十八岁、MySQL 二十八岁、PHP 三十岁。你能遇到的几乎所有问题都被人踩过、写下过、被搜索引擎收录过。
-   **共享主机的兼容性。** 一个 LAMP 应用从 5 块钱一个月的虚拟主机搬到 ECS，代码一行不用改。
-   **可预测的请求路径。** 没有 service mesh、没有 sidecar、没有编排器，一棵进程树一台机器。延迟一上去，`top` 一下大概率能直接看到原因。
-   **极低的运维门槛。** 一台机器，三个服务。需要装在脑子里的东西比 Kubernetes 少一个数量级。

下面几种场景**别**用 LAMP：高扇出 API（用 Nginx + Go/Node/Rust）、长连接事件驱动（大规模 WebSocket 更适合 event loop 而不是 Apache prefork）、团队已经在容器和控制平面上有积累。把 LAMP 放在它擅长的地方用，别把它当默认选项。

# 2. 四层架构

![LAMP 的四层架构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/lamp-%E4%B8%8E%E9%98%BF%E9%87%8C%E4%BA%91%E6%9C%8D%E5%8A%A1%E5%99%A8%E8%AF%A6%E8%A7%A3/fig1_lamp_architecture.png)

上面这张图是整篇文章里最重要的一张。把它内化下来，遇到问题时你会自动从「服务器挂了」切到「哪一层挂了」。

每一层都**owns**一些具体的东西：

-   **Linux** 管进程、文件、socket。如果 `systemctl status apache2` 显示 `inactive`，上面的事情都不用谈。
-   **Apache** 管 HTTP 协议本身，以及把 URL 映射到处理器的那一步。Apache 没起来，80 端口就只是个关着的 socket；起来了但没有 `VirtualHost` 匹配你的 `Host:` 头，请求就会落到默认页。
-   **PHP** 管代码执行。Apache 把一个 `.php` 文件交给它，PHP 解析、运行、返回文本。如果 PHP 缺失或者它的模块没启用，Apache 会高高兴兴地把你的源代码当纯文本送出去——这是个穿着「配置错误」外衣的安全事故。
-   **MySQL** 管持久化。MySQL 挂了，需要数据的 PHP 脚本会抛异常；MySQL 在但凭据不对，同样的脚本会输出空白页。

层与层之间的接口，才是真正会出问题的地方：

| 接口 | 容易出什么问题 | 一行命令就能查 |
| --- | --- | --- |
| Linux -> Apache | 服务没启动、80 端口被占 | `ss -tlnp \| grep ':80'` |
| Apache -> PHP | `php` 模块没启用 | `apache2ctl -M \| grep php` |
| PHP -> MySQL | 扩展没装、host 或 socket 错 | `php -r "var_dump(extension_loaded('mysqli'));"` |
| MySQL -> 磁盘 | 数据目录权限、磁盘满 | `journalctl -u mysql -n 50` |

把这四条命令背下来，你之后基本不需要再去 Stack Overflow 翻 LAMP 的问题。

# 3. 阿里云 ECS 实例长什么样

![阿里云 ECS 实例的解剖图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/lamp-%E4%B8%8E%E9%98%BF%E9%87%8C%E4%BA%91%E6%9C%8D%E5%8A%A1%E5%99%A8%E8%AF%A6%E8%A7%A3/fig2_aliyun_ecs_overview.png)

装东西之前先把机器选对。控制台上密密麻麻几百个选项，但要跑一个起步级的 LAMP 站，真正需要决定的只有四件事：

**地域和可用区。** 地域是城市（杭州、北京、新加坡），可用区是这个城市里的某个机房。用户访问的延迟由地域决定，扛不扛得住一个机房挂掉由可用区决定。单机 LAMP 只能选一个可用区——别在一台机器上假装做多 AZ 高可用，没有意义。

**实例规格族。** 命名是 `<族><代>.<规格>`。对一个公网 LAMP 站来说：

-   `g7.large`（2 vCPU / 8 GiB）是稳妥的默认值——CPU 和内存均衡。
-   `c7.large` 适合工作负载主要是 PHP（CPU bound）、数据库不大的场景。
-   `r7.large` 适合读多、靠 MySQL 缓冲池命中率取胜的场景。
-   `t6` 突发型实例价格只有 `g7` 的一小半，跑一个低流量博客绰绰有余——前提是你理解 CPU 积分会用完。

**磁盘。** ESSD PL1 比基础的云盘强太多。IOPS 从 ~1000 跳到 5000，对应到管理后台的体验就是「卡」和「不卡」的差别，而小容量上的价格差很小。系统盘 40 GiB 够用，数据库会长大就单挂一块数据盘。

**公网 IP。** 直接用实例自带的公网 IP（便宜，但绑死在实例上，释放就没了）或者绑一个弹性公网 IP（EIP，可在实例间漂移）。任何你日后可能重建的环境，都付那点 EIP 的费用。

就这四件事。剩下那一长串高级特性，新手阶段可以无视。

# 4. 网络配置：最容易把人坑住的地方

阿里云论坛上每一条「我的服务器访问不了」，根因都落在这条链路的某一跳：

```
你的笔记本 ---公网---> [安全组] ---> [系统防火墙] ---> [监听 socket] ---> Apache
```

这四道关你必须每一道都开通，否则就会去诊断错的那一层。

## 4.1 公网 IP

控制台 **实例 -> 你的实例 -> 网络与安全 -> 绑定弹性公网 IP**（或者创建实例时直接分配公网 IP）。把这个 IP 记下来，下面用 `8.134.207.88` 当例子。

## 4.2 安全组规则

安全组是一个**有状态的包过滤器**，它跑在云上，不在你的操作系统里。它的判定**早于**任何到达实例的包，所以系统防火墙说什么都没用，安全组说不行就是不行。控制台 **安全组 -> 配置规则 -> 入方向**。

一个公网 LAMP 服务器的安全起步规则集：

| 协议 | 端口范围 | 源地址 | 用途 | 备注 |
| --- | --- | --- | --- | --- |
| TCP | 22/22 | 你家的公网 IP/32 | SSH | 永远别填 `0.0.0.0/0`。`curl ifconfig.me` 拿自己的 IP。 |
| TCP | 80/80 | 0.0.0.0/0 | HTTP | 只用来 301 跳到 HTTPS。 |
| TCP | 443/443 | 0.0.0.0/0 | HTTPS | 真正面向公网的端口。 |
| TCP | 3306/3306 | （关闭） | MySQL | **永远别开。** 走 SSH 隧道。 |
| ICMP | -1/-1 | 0.0.0.0/0 | Ping | 可选，方便监控。 |

如果你确实要给某个开发同学远程连 MySQL，加他单独那个 IP，别加全网。更优雅的办法是直接走 SSH 隧道：

```bash
# 在自己的笔记本上建隧道，而不是去开 3306
ssh -L 33306:127.0.0.1:3306 user@8.134.207.88
# 然后本地连 127.0.0.1:33306
```

跟在安全组里开 3306 比，隧道的好处是：

-   复用你已有的 SSH key，不用多管理一套凭据；
-   只在隧道开着的时候才暴露 DB；
-   永远不会出现在 shodan 的扫描结果里。

## 4.3 操作系统防火墙

云安全组是必要不充分的——未来某个运维同学可能为了「调试方便」把安全组放开，你的第二道防线就是操作系统防火墙。

Ubuntu / Debian：

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status numbered
```

CentOS / Alibaba Cloud Linux（firewalld）：

```bash
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
sudo firewall-cmd --list-all
```

## 4.4 一跳一跳验

访问不了的时候，按下面这个**严格顺序**排查。乱了顺序，你会浪费两个小时去查错的那一层。

```bash
# 第 1 步：公网 IP 通不通
ping 8.134.207.88
# 不通 -> 安全组 ICMP 没开，或实例已经停掉

# 第 2 步：操作系统上有进程在监听吗
ss -tlnp | grep -E ':80|:443|:22'
# 没看到 -> 服务没起

# 第 3 步：操作系统防火墙放行了吗
sudo ufw status            # 或者 sudo firewall-cmd --list-all

# 第 4 步：本机能 curl 通 Apache 吗
curl -I http://127.0.0.1
# 200 OK -> 问题在 Apache 上游（防火墙 / 安全组）
```

# 5. 一个请求从头到尾走过的路

![一个 HTTP 请求穿过整个栈的过程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/lamp-%E4%B8%8E%E9%98%BF%E9%87%8C%E4%BA%91%E6%9C%8D%E5%8A%A1%E5%99%A8%E8%AF%A6%E8%A7%A3/fig3_request_flow.png)

请求最终送到 Apache 时发生的事：

1.  浏览器对 `8.134.207.88:80` 发起 TCP 连接。
2.  阿里云安全组放行 SYN（80 端口对 0.0.0.0/0 的规则）。
3.  内核把这个连接交给监听者——`apache2`。
4.  Apache 解析请求行 `GET /index.php HTTP/1.1`，遍历 `VirtualHost` 配置找到 `ServerName` 匹配 `Host:` 头的那一条，再用这条 vhost 的 `DocumentRoot` 去解析 `/index.php`。
5.  `mod_php` 处理器命中 `.php`，Apache 调用内嵌的 PHP 解释器（如果用 FPM，则是打开 unix socket 把请求转过去）。
6.  PHP 脚本开始跑，第一句多半是 `new mysqli('localhost', ...)` 或者 `new PDO('mysql:host=localhost;...')`。PHP 对 `127.0.0.1:3306` 发起 TCP 连接（在 Debian/Ubuntu 上更可能是走 unix socket `/var/run/mysqld/mysqld.sock`）。
7.  MySQL 验证用户、解析 SQL，命中 InnoDB 缓冲池（没命中就读盘），返回结果集。
8.  PHP 把结果渲染成 HTML，交回 Apache，由 Apache 写回 socket。

图里在每一跳下面标了最常见的故障模式。最高频的三种：

-   **PHP 源码直接显示在浏览器里。** Apache 把文件送出去了但没调用 PHP，模块没加载。
-   **装完之后空白页。** PHP 错误被静默掉了，脚本崩了——去 `/var/log/apache2/error.log` 找，不要在浏览器里找。
-   **偶发的「Connection refused」。** MySQL 连接数打满了，或者 OOM killer 把 `mysqld` 杀了。看 `dmesg` 和 `mysql.err`。

# 6. 在 Ubuntu 上装

第一步，先确认机器上没有别的 web 服务器或数据库在跑：

```bash
sudo systemctl status apache2 nginx mysql mariadb 2>/dev/null \
  | grep -E 'Active|service'
# 不要的就停掉、屏蔽掉
sudo systemctl disable --now nginx
```

安装顺序很重要：先 Apache，再 MySQL，最后 PHP。PHP 的包会顺手把 Apache 模块拉进来并启用——前提是 Apache 已经在那儿。

## 6.1 Apache

```bash
sudo apt update
sudo apt install -y apache2
sudo systemctl enable --now apache2
```

浏览器访问 `http://你的公网IP/`，应该看到 Apache2 Ubuntu Default Page。如果看不到，按 4.4 节的四步**严格顺序**排查。

实际会编辑的目录：

| 路径 | 放什么 |
| --- | --- |
| `/etc/apache2/apache2.conf` | 全局配置——基本不直接改 |
| `/etc/apache2/sites-available/*.conf` | 虚拟主机定义 |
| `/etc/apache2/sites-enabled/` | 软链；`a2ensite` / `a2dissite` 管理 |
| `/etc/apache2/mods-available/*.{load,conf}` | 模块配置——`a2enmod` 管理 |
| `/var/www/html/` | 默认 DocumentRoot |
| `/var/log/apache2/{access,error}.log` | 出问题第一时间看的地方 |

调试期可以临时把日志级别调高，调完再调回去：

```bash
sudo sed -i 's/^LogLevel warn/LogLevel info/' /etc/apache2/apache2.conf
sudo systemctl reload apache2
```

## 6.2 MySQL

```bash
sudo apt install -y mysql-server
sudo systemctl enable --now mysql
sudo mysql_secure_installation
```

`mysql_secure_installation` 会问五个问题。建议的答案：

1.  **VALIDATE PASSWORD plugin：** yes，强度选 2（strong）。
2.  **设置 root 密码：** 用密码管理器生成的 16 位以上密码，存好。
3.  **删除匿名用户：** yes。
4.  **禁止 root 远程登录：** yes——你之后会用 SSH 隧道。
5.  **删除 test 库：** yes。

然后验：

```bash
sudo systemctl status mysql
sudo mysql -e "SELECT version(), @@hostname;"
```

### `caching_sha2_password` 这个坑

MySQL 8.0 把默认认证插件从 `mysql_native_password` 改成了 `caching_sha2_password`。老一点的 PHP `mysqli` 驱动、PHP 自带的 `mysql` 扩展、还有不少 CMS 的安装程序不会说新协议，会报 `The server requested authentication method unknown to the client`。今天最正确的做法是升级驱动；现实里没条件升级时，**只针对应用账号**切回老插件：

```sql
ALTER USER 'discuz_user'@'localhost'
  IDENTIFIED WITH mysql_native_password BY 'StrongPassword123!';
FLUSH PRIVILEGES;
```

只对应用账号做，**永远别**这样削弱 root。

### 一份起步可用的 my.cnf 调优

MySQL 8 出厂的缓冲池小得可怜。对哪怕只是稍微忙一点的站点来说，这是性能差距最大的单一旋钮：

```ini
# /etc/mysql/mysql.conf.d/zz-tuning.cnf
[mysqld]
innodb_buffer_pool_size = 4G          # 数据库专机大约用一半内存
innodb_log_file_size    = 512M
innodb_flush_log_at_trx_commit = 1    # 设 2 可以以丢 1 秒数据为代价换速度
max_connections = 200
character-set-server   = utf8mb4
collation-server       = utf8mb4_unicode_ci
```

改完重启 MySQL。缓冲池一项就能决定「每条 query 都打盘」和「热数据全在 RAM」的差别。

## 6.3 PHP

```bash
sudo apt install -y php libapache2-mod-php php-mysql \
                    php-curl php-gd php-mbstring php-xml php-zip php-intl
sudo systemctl restart apache2
```

验 Apache 和 PHP 之间的桥：

```bash
echo '<?php phpinfo();' | sudo tee /var/www/html/info.php
curl -s http://127.0.0.1/info.php | head -n 5
```

应该看到 HTML，不是 PHP 源码。看到源码的话，说明 `php` 模块没启用：

```bash
ls /etc/apache2/mods-enabled/php*.load   # 有吗？
sudo a2enmod php8.1                      # 跟你装的版本对上
sudo systemctl restart apache2
```

**测完立刻把 `info.php` 删掉**——它会泄露整份 PHP 配置，包括加载了哪些扩展、文件路径、`disable_functions`。攻击者扫描器第一个搜的就是它。

```bash
sudo rm /var/www/html/info.php
```

# 7. 纵深防御：把公网面收紧

![一台公网 LAMP 服务器的纵深防御](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/lamp-%E4%B8%8E%E9%98%BF%E9%87%8C%E4%BA%91%E6%9C%8D%E5%8A%A1%E5%99%A8%E8%AF%A6%E8%A7%A3/fig4_security_setup.png)

一台公网 LAMP 用默认配置上线，几分钟之内就会被自动扫描器开始探。把安全当作五圈同心圆——任何一圈被破，外面下一圈还能撑一段时间。

## 7.1 安全组——最外圈

第 4 节讲过了。原则是：安全组应该让操作系统防火墙看起来多余，操作系统防火墙也应该让安全组看起来多余。任何一道单独都不够。

## 7.2 操作系统加固

```bash
# 系统补丁——开启自动安全更新
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# 禁用 SSH 密码登录（确认密钥能登之后再做）
sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# 装 fail2ban 减缓爆破
sudo apt install -y fail2ban
sudo systemctl enable --now fail2ban
```

## 7.3 用 Let's Encrypt 上 HTTPS

域名 A 记录指向你的公网 IP 之后，签证书就是两条命令：

```bash
sudo apt install -y certbot python3-certbot-apache
sudo certbot --apache -d example.com -d www.example.com \
             --redirect --agree-tos -m you@example.com
```

`certbot --apache` 会写一份 443 的 vhost、启用 `mod_ssl` 和 `mod_rewrite`、加一条 80 -> 443 的 301。它还会装一个 systemd timer 在到期前自动续期，验一下：

```bash
sudo systemctl list-timers | grep certbot
sudo certbot renew --dry-run
```

现代的 TLS 配置不只是「打开 TLS」，还要把不安全的部分关掉。在 `/etc/apache2/sites-available/example.com-le-ssl.conf` 里加上：

```apache
SSLProtocol             all -SSLv3 -TLSv1 -TLSv1.1
SSLCipherSuite          HIGH:!aNULL:!MD5:!3DES
SSLHonorCipherOrder     on
Header always set Strict-Transport-Security "max-age=63072000"
```

## 7.4 MySQL 加固

-   绑定到 `127.0.0.1`（新版包默认就是，去 `/etc/mysql/mysql.conf.d/mysqld.cnf` 确认一下）。
-   **每个应用一个数据库账号**，`GRANT` 范围限定到那个库。
-   永远不要 `GRANT ALL ... TO root@'%'`。
-   敏感数据的备份要加密落盘。

## 7.5 应用层卫生

-   能用 `php-fpm` 就别用 `mod_php`——把 PHP 故障从 Apache 进程树里隔离出去。
-   生产环境的 `/etc/php/8.1/apache2/php.ini` 里 `expose_php = Off`、`display_errors = Off`。
-   不论部署什么框架，都要订阅它的安全公告。LAMP 服务器被入侵的最大单一来源就是 CMS 的 CVE。

# 8. 端到端部署：Discuz!

拿 Discuz! 当例子，是因为它把一个新装的 LAMP 的薄弱环节都敲打了一遍：文件权限、多个可写目录、MySQL 用户创建、PHP 扩展依赖、还有一个 web 安装器把这些都重新校验一次。

## 8.1 下载

```bash
cd /var/www/html
sudo wget https://download.comsenz.com/DiscuzX/3.4/Discuz_X3.4_SC_UTF8.zip
sudo apt install -y unzip
sudo unzip -q Discuz_X3.4_SC_UTF8.zip
sudo mv upload/* upload/.htaccess . 2>/dev/null || sudo mv upload/* .
sudo rm -rf upload Discuz_X3.4_SC_UTF8.zip readme.txt utility/
```

## 8.2 权限——人人都搞错的地方

Apache 跑在 `www-data`（Ubuntu）或 `apache`（CentOS）下。唯一的规则：**Apache 跑的那个用户必须 owns PHP 需要写入的所有文件，且仅此而已**。

```bash
# 基线：所有文件可读，归 web 用户所有
sudo chown -R www-data:www-data /var/www/html
sudo find /var/www/html -type d -exec chmod 755 {} \;
sudo find /var/www/html -type f -exec chmod 644 {} \;

# Discuz 需要可写的目录（安装器会校验）
for d in data config uc_server/data uc_client/data; do
  sudo chmod -R 775 /var/www/html/$d
done
```

注意是 `775`，**不是** `777`。`www-data` 已经是属主了，`775` 让属主能写，同时只给 group 加写权限。`chmod 777` 是江湖偏方，不是建议——它让系统上**任何**用户都能改你的应用文件，在共享服务器上就是一条提权路径。

## 8.3 数据库账号

```bash
sudo mysql -e "
  CREATE DATABASE discuz CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
  CREATE USER 'discuz_user'@'localhost' IDENTIFIED BY 'StrongPassword123!';
  GRANT ALL PRIVILEGES ON discuz.* TO 'discuz_user'@'localhost';
  FLUSH PRIVILEGES;
"
```

两点注意：

-   `discuz.*`——授权范围是单一数据库。Discuz 即使被打穿，攻击者也读不到你别的应用的表。
-   `'discuz_user'@'localhost'`——主机部分是身份的一部分。同名用户从不同主机来是不同用户。走 unix socket 算 `'localhost'`，TCP 到 `127.0.0.1` 算 `'127.0.0.1'`。如果 `mysql_secure_installation` 之后这两个不等价，两个都要授权。

## 8.4 跑安装器

访问 `http://你的公网IP/install/`。三件事会发生：

1.  **环境检查**——PHP 版本、GD、mbstring、mysqli。缺哪个就 `sudo apt install -y php-<扩展> && sudo systemctl reload apache2`。
2.  **权限检查**——`data/`、`config/`、`uc_server/data/`、`uc_client/data/` 旁边都应该是绿色的勾。没勾就回 8.2 重新检查。
3.  **数据库信息**——host 填 `localhost`，name `discuz`，user `discuz_user`，密码用上面设的。

装完之后：

```bash
sudo rm -rf /var/www/html/install
# 不再需要可写的目录可以收紧回去
sudo chmod -R 755 /var/www/html/config
```

# 9. 90% 的人会撞上的 5 个故障

## 故障 1 ——「Connection refused」

**含义：** 你和 Apache 之间有什么东西在丢 SYN，或者 Apache 没在监听。

```bash
# 在服务器本地执行——能通就说明 Apache 没问题，问题在网络
curl -I http://127.0.0.1

# 实际在监听的有谁？
sudo ss -tlnp | grep -E ':80|:443'

# 系统防火墙怎么说？
sudo ufw status
```

`127.0.0.1` 通但公网 IP 不通，说明系统这一侧是好的——去云控制台看安全组。

## 故障 2 ——「403 Forbidden」或「Index of /」

**含义：** Apache 把目录返回了，但里面没有 index 文件，或者读不到。

```bash
ls -l /var/www/html/                       # index.php / index.html 在吗
sudo -u www-data cat /var/www/html/index.php # web 用户能读吗
grep -r DirectoryIndex /etc/apache2/
```

修复方案 99% 是 `chown -R www-data:www-data /var/www/html`——你以 root 身份 `wget` 了某个东西，web 用户读不动。

## 故障 3 —— 浏览器里看到 PHP 源码

**含义：** Apache 把 `.php` 当成静态文件返回了，PHP 处理器没注册。

```bash
apache2ctl -M | grep php_module    # 没输出？PHP 模块没启用
sudo a2enmod php8.1                # 跟你的版本对上
sudo systemctl restart apache2
curl -s http://127.0.0.1/info.php | head -n 1   # 应该是 HTML，不是 <?php
```

这是个安全事故，不只是配置问题——别让机器在这个状态下挂在公网上。

## 故障 4 ——「Can't connect to MySQL server on 'localhost'」

**含义：** MySQL 没跑、socket 路径变了、或凭据错了。

```bash
sudo systemctl status mysql
sudo journalctl -u mysql -n 100 --no-pager
mysql -u discuz_user -p -h 127.0.0.1 -e 'SELECT 1'
```

小规格实例上常见原因：MySQL 被 OOM-killed。`dmesg | tail -50` 会显示 `Killed process ... mysqld`。要么把 `innodb_buffer_pool_size` 调小，要么换更大的实例。

## 故障 5 —— Discuz 提示「目录不可写」

**含义：** web 用户写不了那四个目录之一。

```bash
sudo -u www-data touch /var/www/html/data/.write_test
# Permission denied？那就：
sudo chown -R www-data:www-data /var/www/html/{data,config,uc_server/data,uc_client/data}
sudo chmod -R 775 /var/www/html/{data,config,uc_server/data,uc_client/data}
```

忍住别 `chmod -R 777 /var/www`。它能解决问题，但日后会反过来咬你。

# 10. 上生产前要做完的几件事

## 10.1 虚拟主机

只要你有一个以上的站点，就别再把所有东西堆在 `/var/www/html/` 里。每个站点一个 `/var/www/<站点>/` 目录、一个 vhost 文件，结构会清晰很多。

```apache
# /etc/apache2/sites-available/blog.example.com.conf
<VirtualHost *:443>
    ServerName  blog.example.com
    ServerAlias www.blog.example.com

    DocumentRoot /var/www/blog.example.com
    <Directory /var/www/blog.example.com>
        Options -Indexes +FollowSymLinks
        AllowOverride All
        Require all granted
    </Directory>

    SSLEngine on
    SSLCertificateFile      /etc/letsencrypt/live/blog.example.com/fullchain.pem
    SSLCertificateKeyFile   /etc/letsencrypt/live/blog.example.com/privkey.pem

    ErrorLog  ${APACHE_LOG_DIR}/blog.example.com.error.log
    CustomLog ${APACHE_LOG_DIR}/blog.example.com.access.log combined
</VirtualHost>
```

```bash
sudo a2ensite blog.example.com
sudo apache2ctl configtest && sudo systemctl reload apache2
```

`configtest` 在 `reload` 之前执行，是「平滑切换」和「打错括号宕机五分钟」之间的差别。

## 10.2 真正能恢复的备份

没演练过恢复的备份不是备份。最低限度：

```bash
# /usr/local/bin/db-backup.sh
#!/usr/bin/env bash
set -euo pipefail
DEST=/var/backups/mysql
DATE=$(date +%F_%H%M)
mkdir -p "$DEST"
mysqldump --single-transaction --routines --events \
          --databases discuz \
        | gzip > "$DEST/discuz_$DATE.sql.gz"
# 保留 14 天
find "$DEST" -name 'discuz_*.sql.gz' -mtime +14 -delete
```

```cron
# crontab -e
0 3 * * * /usr/local/bin/db-backup.sh
```

每周再同步一份到 OSS，免得磁盘挂了备份也一起没：

```bash
ossutil cp -r /var/backups/mysql/ oss://mybucket/db-backups/$(hostname)/
```

每个月在另一台机器上跑一次 `gunzip < some_backup.sql.gz | mysql -u root -p test_restore` 并核对行数。第一次演练一定会让你长见识。

## 10.3 可观测性

阿里云的 Cloud Monitor agent 默认就给你 CPU、内存、磁盘、带宽。值得自己再加两个信号：

-   Apache `mod_status` 暴露在 `127.0.0.1:80/server-status`——QPS、忙的 worker 数、慢请求。
-   MySQL `performance_schema` 找慢查询：`SELECT digest_text, count_star, avg_timer_wait FROM events_statements_summary_by_digest ORDER BY sum_timer_wait DESC LIMIT 10`。

每��花五分钟看一眼，能在容量真出事之前几周就发现端倪。

# 11. 一个应用，两种拓扑

![LAMP 站点的两种部署拓扑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/standalone/lamp-%E4%B8%8E%E9%98%BF%E9%87%8C%E4%BA%91%E6%9C%8D%E5%8A%A1%E5%99%A8%E8%AF%A6%E8%A7%A3/fig5_deployment_topology.png)

几乎每个能跑起来的 LAMP 站，最终都会撞到单机拓扑的天花板，然后要决定是纵向扩（更大的 ECS）还是横向拆（拆三层）。上图就是这个决定的两个目的地。

**继续单机的条件**：峰值流量塞得进一台机器，数据库小到缓冲池能盖住热数据，团队就一个工程师。PHP 和 MySQL 之间走 localhost 的连接比任何网络调用都快，运维面只是一个操作系统要打补丁。

**该拆三层的信号**：需要横向扩容（SLB 后挂多个 PHP worker），需要高可用（RDS 直接给你主备），或者你为单台 ECS 付的钱已经超过两台小 ECS + 一个 RDS。阿里云上经典三层：

-   **SLB** 终结 TLS、把请求分发给 web 层。
-   **N 台 ECS** 跑 Apache + PHP，全部无状态（session 放 Redis，上传放 OSS）。
-   **RDS for MySQL** 作为唯一的事实来源。

成本大致变成 3 倍，故障面从「一台机器」变成「多台机器加一段网络」，运维确实更难。别因为图画得好看就迁——单机真的扛不住了再迁。

# 12. 源码编译 MySQL（进阶）

通常你不需要这么干。除非有具体理由——发行版省了某个编译选项、合规要求锁死某个版本、上游还没 merge 的补丁——否则用包管理器。源码编译的代价是真的：编译几个小时、没自动安全更新、CVE 全靠你自己跟。

如果一定要做，CentOS 上编译 MySQL 5.6 的标准咒语：

```bash
sudo yum install -y gcc gcc-c++ cmake bison \
                    libaio-devel ncurses-devel zlib-devel openssl-devel

cd /usr/local
sudo mkdir software-mysql && cd software-mysql
sudo wget https://repo.huaweicloud.com/mysql/Downloads/MySQL-5.6/mysql-5.6.49.tar.gz
sudo tar -xzf mysql-5.6.49.tar.gz
cd mysql-5.6.49

sudo cmake . \
  -DCMAKE_INSTALL_PREFIX=/usr/local/mysql \
  -DMYSQL_DATADIR=/usr/local/mysql/data \
  -DENABLE_LOCAL_INFILE=1 \
  -DWITH_INNOBASE_STORAGE_ENGINE=1 \
  -DMYSQL_TCP_PORT=3306 \
  -DDEFAULT_CHARSET=utf8mb4 \
  -DDEFAULT_COLLATION=utf8mb4_general_ci \
  -DWITH_EXTRA_CHARSETS=all \
  -DMYSQL_USER=mysql

sudo make -j$(nproc)         # 2 vCPU 上 1-3 小时
sudo make install

sudo useradd -r -s /sbin/nologin mysql
sudo chown -R mysql:mysql /usr/local/mysql
cd /usr/local/mysql
sudo ./scripts/mysql_install_db --user=mysql

sudo cp support-files/mysql.server /etc/init.d/mysql
sudo systemctl enable --now mysql
```

几乎每次都会撞到的两个坑：

1.  **`Could not find OpenSSL`**——你少装了 `openssl-devel`。修复方式：装上之后**把整个编译目录删掉重新解压**再编。`cmake` 会缓存中间状态，半截的目录不会重新发现新装的头文件。
2.  **编译过程被 OOM**——2 GiB 的实例上 `make -j$(nproc)` 一定会被杀。改成 `-j2`，并且开始之前先加 2 GiB 的 swap。

装完别忘了同样要走 6.2 节的 `mysql_secure_installation` 和 `my.cnf` 调优——源码编译版本不会替你配好。

# 13. 几个真实场景

## 场景 A —— 把 WordPress 从虚拟主机迁过来

被反复验证过的菜谱：

```bash
# 在老主机上（cPanel / phpMyAdmin / SSH）
mysqldump --single-transaction wordpress > wp.sql
tar czf wp-files.tar.gz public_html/

# 通过 scp / OSS 搬到 ECS 上
scp wp.sql wp-files.tar.gz user@8.134.207.88:/tmp/

# 在 ECS 上
cd /var/www/html
sudo tar xzf /tmp/wp-files.tar.gz --strip-components=1
sudo mysql -e "CREATE DATABASE wordpress; \
               CREATE USER 'wp'@'localhost' IDENTIFIED BY '...'; \
               GRANT ALL ON wordpress.* TO 'wp'@'localhost';"
sudo mysql wordpress < /tmp/wp.sql

# 改 wp-config.php 里的数据库账号
# 把站点 URL 同时改两处：
sudo mysql wordpress -e "UPDATE wp_options SET option_value='https://example.com' \
                         WHERE option_name IN ('siteurl','home');"

# 权限
sudo chown -R www-data:www-data /var/www/html
sudo find /var/www/html -type d -exec chmod 755 {} \;
sudo find /var/www/html -type f -exec chmod 644 {} \;

# DNS A 记录改成 8.134.207.88，TTL 调小方便切换
```

唯一的雷几乎永远是权限——虚拟主机上一切都归你这个用户，到了 ECS 上一切归 `www-data`，上传会一直失败直到你修对。

## 场景 B —— 一台机器跑两个 PHP 版本

老插件要 PHP 5.6，新应用要 8.1。用 `php-fpm` 装两套版本，按 vhost 路由：

```bash
sudo add-apt-repository ppa:ondrej/php
sudo apt update
sudo apt install -y php5.6-fpm php8.1-fpm
```

```apache
<VirtualHost *:443>
    ServerName legacy.example.com
    DocumentRoot /var/www/legacy
    <FilesMatch \.php$>
        SetHandler "proxy:unix:/run/php/php5.6-fpm.sock|fcgi://localhost"
    </FilesMatch>
</VirtualHost>

<VirtualHost *:443>
    ServerName modern.example.com
    DocumentRoot /var/www/modern
    <FilesMatch \.php$>
        SetHandler "proxy:unix:/run/php/php8.1-fpm.sock|fcgi://localhost"
    </FilesMatch>
</VirtualHost>
```

这也是顺手摆脱 `mod_php` 的好时机。`php-fpm` 把 PHP 跑在自己的进程池里，有自己的用户、自己的资源限制、自己的崩溃恢复——PHP 里漏内存不再会把 Apache 一起拉下水。

## 场景 C —— 流量上来之后偶发 502

很常见的连环反应：流量翻倍，站点开始对 5% 的请求返回 502。原因链几乎总是同一个：

1.  Apache prefork 打满 `MaxRequestWorkers`，新连接开始排队。
2.  PHP-FPM 打满 `pm.max_children`，Apache 从 FPM socket 拿到的就是 502。
3.  MySQL 打满 `max_connections`，PHP-FPM worker 阻塞在拿连接的位置直到超时。

修法是把每一层的 worker pool 都调到比上一层略多。4 vCPU / 16 GiB 实例上一个起步参考：

```ini
# /etc/apache2/mods-available/mpm_event.conf
StartServers             4
MinSpareThreads         50
MaxSpareThreads        100
ThreadsPerChild         25
MaxRequestWorkers      300

# /etc/php/8.1/fpm/pool.d/www.conf
pm = dynamic
pm.max_children      = 60
pm.start_servers     = 10
pm.min_spare_servers = 5
pm.max_spare_servers = 20

# my.cnf
max_connections = 200
```

数字不是魔法，原则才是：每一层的 worker pool 必须能消化上面那一层送下来的并发，且不会等太久。

# 14. 总结

阿里云 ECS 上的 LAMP 浓缩成一份 5 步清单：

1.  **端口开对**——安全组、再操作系统防火墙、再一跳一跳验通。
2.  **按顺序装**——Apache、MySQL、PHP，每装一个先验过再装下一个。
3.  **每层验通**——Apache 出 HTML，PHP 跑得了，MySQL 连得上。三条命令，每次都跑。
4.  **权限给得有意识**——`www-data` owns 该可写的部分，绝不 `chmod 777`。
5.  **部署应用**——Discuz、WordPress、自己写的 PHP，套路都一样。

接下来值得做的事：

-   用 Let's Encrypt 配上 HTTPS，并打开 HSTS。
-   把 `mysqldump -> OSS` 跑起来，并且**真的从备份恢复一次**。
-   花一个小时读自己的 access log——比任何博客文章都让你了解你的流量和你的攻击者。
-   单机扛不住了再拆 SLB + ECS + RDS，别死命扩同一台机器。

延伸阅读：

-   Apache HTTP Server 文档 —— <https://httpd.apache.org/docs/>
-   MySQL 8.0 参考手册 —— <https://dev.mysql.com/doc/refman/8.0/en/>
-   PHP 手册 —— <https://www.php.net/manual/en/>
-   阿里云 ECS 用户指南 —— <https://help.aliyun.com/zh/ecs/>
-   Let's Encrypt 与 Certbot —— <https://certbot.eff.org/instructions>
