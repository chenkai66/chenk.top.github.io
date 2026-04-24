---
title: "LAMP Stack on Alibaba Cloud ECS: From Fresh Instance to Production-Ready Web Server"
date: 2025-01-05 09:00:00
tags:
  - Cloud Computing
  - Linux
  - Web Server
categories: Tutorial
lang: en
description: "Set up a LAMP stack (Linux, Apache, MySQL, PHP) on Alibaba Cloud ECS. Covers security groups, service installation, Discuz deployment, source compilation, hardening and three-tier scale-out."
---

You have a fresh ECS instance and SSH access. Your goal is a public website running Apache, PHP and MySQL. Between you and that goal sit three classes of problems that catch every beginner the first time:

1.  **Network reachability** -- packets are silently dropped at the cloud security group, the OS firewall, or the listening socket, and the symptom is the same in all three cases: nothing happens.
2.  **Service wiring** -- Apache, PHP and MySQL are three separate processes that have to find each other through file extensions, Unix sockets and TCP ports. Each interface has its own failure mode.
3.  **Identity and permissions** -- Apache runs as `www-data`, MySQL runs as `mysql`, files are owned by `root` after `wget`. The wrong combination produces 403, "Access denied", or `chmod 777` desperation.

This guide walks through all of them in the order you actually hit them on day one, then keeps going into the things that show up on day thirty: TLS, virtual hosts, backups, source compilation, and when to stop running everything on a single box.

## What you will be able to do after reading

-   Build a mental model of how an HTTP request travels through Linux, Apache, PHP and MySQL, and predict where it will break.
-   Configure Aliyun networking from the security group inwards, with a real defence-in-depth model rather than `0.0.0.0/0` everywhere.
-   Install, verify and harden each LAMP component on Ubuntu (the steps for CentOS / Alibaba Cloud Linux are called out alongside).
-   Deploy a non-trivial application end-to-end (Discuz!), including the file-permission and database-account work that the docs gloss over.
-   Diagnose the five failures that account for ~90% of "my LAMP doesn't work" tickets.
-   Decide when to stay on a single ECS, and when to split into SLB + ECS + RDS.

## Prerequisites

-   An Alibaba Cloud ECS instance, Ubuntu 22.04 LTS or Alibaba Cloud Linux 3 / CentOS 7+.
-   SSH access from your laptop using a key (not a password).
-   Comfort with the Linux command line: `ls`, `cd`, `cat`, `systemctl`, `sudo`.
-   A domain name is optional but nice to have for the TLS section.

---

# 1. Why LAMP still earns its place

LAMP -- **L**inux + **A**pache + **M**ySQL + **P**HP -- has been declared dead in every web framework cycle since 2010 and refuses to oblige. The reason is not nostalgia, it is fitness for purpose. For content sites, CMS platforms (WordPress, Discuz, Drupal, MediaWiki), customer portals, internal tools and a long tail of small SaaS backends, LAMP is the **most cost-effective, best-documented and lowest-maintenance** way to put dynamic web pages in front of users.

What you actually get for free with LAMP that newer stacks make you reassemble:

-   **A mature ecosystem.** Apache is twenty-eight years old, MySQL twenty-eight, PHP thirty. Almost every problem you can have has been hit, written up and indexed.
-   **Shared-hosting parity.** A LAMP app moves between a $5/month shared host and an ECS without a code change.
-   **A predictable request path.** No service mesh, no sidecar, no orchestrator -- one process tree on one machine. When latency rises you can `top` your way to the answer.
-   **A low operational floor.** One server, three services. The cognitive load is a fraction of Kubernetes.

LAMP is **not** the right answer when your workload is high-fanout APIs (use Nginx + Go/Node/Rust), event-driven and connection-heavy (websockets at scale prefer event loops over Apache prefork), or when your team has already invested in containers and a control plane. Pick LAMP for what it is good at, not as a default.

# 2. The four-layer architecture

![The four-layer LAMP stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lamp-on-ecs/fig1_lamp_architecture.png)

The diagram above is the single most important picture in this guide. Internalise it and most operational problems become "which layer is broken" instead of "the server doesn't work".

Each layer **owns** something specific:

-   **Linux** owns processes, files and sockets. If `systemctl status apache2` says `inactive`, the rest does not matter.
-   **Apache** owns the HTTP wire format and the mapping from URL to handler. If Apache is not loaded, port 80 is just a closed socket; if it is loaded but no `VirtualHost` matches your `Host:` header, you fall through to the default page.
-   **PHP** owns code execution. Apache hands it a `.php` file; PHP parses it, runs it, returns text. If PHP is missing or its module is not enabled, Apache happily serves your source code as plain text -- a security incident dressed as a misconfiguration.
-   **MySQL** owns durability. If MySQL is down, PHP scripts that need data raise exceptions; if MySQL is up but the credentials are wrong, the same scripts produce blank pages.

The interfaces between the layers are the parts that fail:

| Interface | What can go wrong | Fast check |
| --- | --- | --- |
| Linux -> Apache | service not started, port 80 in use | `ss -tlnp \| grep ':80'` |
| Apache -> PHP | `php` module not enabled | `apache2ctl -M \| grep php` |
| PHP -> MySQL | extension missing, wrong host/socket | `php -r "var_dump(extension_loaded('mysqli'));"` |
| MySQL -> disk | data directory permissions, full disk | `journalctl -u mysql -n 50` |

Memorise these four checks and you will resolve most LAMP issues without ever opening Stack Overflow.

# 3. Anatomy of an Aliyun ECS instance

![Anatomy of an Alibaba Cloud ECS instance](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lamp-on-ecs/fig2_aliyun_ecs_overview.png)

Before installing anything, pick the right instance. The console shows hundreds of options; in practice for a starter LAMP server you decide on four things:

**Region and zone.** A region is a city (Hangzhou, Beijing, Singapore); a zone is a data centre inside that city. Latency to your users is set by the region; resilience to one DC failure is set by the zone. For a single-instance LAMP you only pick one zone -- there is no point pretending to be multi-AZ on one machine.

**Instance family.** The naming is `<family><generation>.<size>`. For a public LAMP site:

-   `g7.large` (2 vCPU / 8 GiB) is the safe default -- balanced compute and memory.
-   `c7.large` if your workload is mostly PHP (CPU-bound) and your DB is small.
-   `r7.large` if your workload is read-heavy and you can win by caching aggressively in MySQL's buffer pool.
-   `t6` burstable instances cost a fraction of `g7` and are fine for a low-traffic blog -- as long as you understand CPU credits run out.

**Disk.** Choose ESSD PL1 over basic cloud disks. The IOPS difference (5000 vs ~1000) is the difference between a snappy admin panel and a slow one, and the price gap on small disks is small. Forty GiB is enough for the OS plus a moderate site -- attach a separate data disk if your database will grow past a few gigabytes.

**Public IP.** You can take the public IP that comes with the instance (cheap, but bound to the instance and lost on release) or attach an Elastic IP (EIP) which survives instance changes. For anything you might rebuild, pay the small EIP fee.

That is the entire decision. Skip the long list of features and confirm.

# 4. Networking: the part that traps everyone

Every "I can't reach my server" question on the Aliyun forum has the same root cause -- packets are dropped at one of the four points in the path:

```
client laptop ---internet---> [security group] ---> [OS firewall] ---> [listen socket] ---> Apache
```

You have to open all four, or you will diagnose the wrong layer.

## 4.1 Public IP

In the ECS console, **Instances -> your instance -> Networking -> Bind EIP** (or assign a public IP at create time). Note the address; treat it like a domain name (`8.134.207.88` is the example used below).

## 4.2 Security group rules

The security group is a stateful packet filter that lives in the cloud, not on the OS. It runs **before** anything reaches your instance, so it overrides whatever your OS firewall says. In the console: **Security Groups -> Configure Rules -> Inbound**.

A safe starter rule set for a public LAMP server:

| Protocol | Port range | Source | Purpose | Notes |
| --- | --- | --- | --- | --- |
| TCP | 22/22 | your home IP/32 | SSH | Never `0.0.0.0/0`. Use `curl ifconfig.me` to find your IP. |
| TCP | 80/80 | 0.0.0.0/0 | HTTP | Only as a `301 -> https` redirect. |
| TCP | 443/443 | 0.0.0.0/0 | HTTPS | The only port the public actually talks to. |
| TCP | 3306/3306 | (closed) | MySQL | **Never open.** Reach DB via SSH tunnel. |
| ICMP | -1/-1 | 0.0.0.0/0 | Ping | Optional, useful for monitoring. |

If you genuinely need remote MySQL access for a developer, add their IP only:

```bash
# On your laptop, build a tunnel instead of opening 3306
ssh -L 33306:127.0.0.1:3306 user@8.134.207.88
# Then connect locally to 127.0.0.1:33306
```

Compared to opening 3306 on the security group, the tunnel:

-   reuses your existing SSH key auth (no extra credential),
-   only exposes the DB while the tunnel is up,
-   never appears in shodan scans.

## 4.3 OS-level firewall

The cloud security group is necessary but not sufficient -- a future operator might open everything on the security group "to debug", and your second line of defence is the OS firewall.

On Ubuntu / Debian:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
sudo ufw status numbered
```

On CentOS / Alibaba Cloud Linux (firewalld):

```bash
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
sudo firewall-cmd --list-all
```

## 4.4 Verifying reachability hop by hop

When something does not respond, isolate the failing hop in this exact order. Doing it out of order is how you spend three hours debugging the wrong thing.

```bash
# Step 1 -- can I reach the public IP at all?
ping 8.134.207.88
# fails  -> security group ICMP rule, or instance is stopped

# Step 2 -- is anything listening on the OS?
ss -tlnp | grep -E ':80|:443|:22'
# missing -> service not started

# Step 3 -- does the OS firewall pass it?
sudo ufw status            # or: sudo firewall-cmd --list-all

# Step 4 -- does Apache answer locally?
curl -I http://127.0.0.1
# 200 OK -> the problem is upstream of Apache (firewall / SG)
```

# 5. The request flow, end to end

![How a request travels through the stack](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lamp-on-ecs/fig3_request_flow.png)

When the request finally lands and Apache answers, this is what happens. Knowing this flow turns "the site is broken" into a sequence of yes/no questions.

1.  Browser opens a TCP connection to `8.134.207.88:80`.
2.  Aliyun's security group accepts the SYN (rule for tcp:80 from 0.0.0.0/0).
3.  The kernel hands the connection to whoever is listening -- `apache2`.
4.  Apache parses the request line `GET /index.php HTTP/1.1`, walks its `VirtualHost` config to find one whose `ServerName` matches the `Host:` header, then resolves `/index.php` against that vhost's `DocumentRoot`.
5.  The `mod_php` handler matches `.php` and Apache invokes the embedded PHP interpreter (or, with FPM, opens the unix socket and forwards the request).
6.  The PHP script runs; one of its first statements is usually `new mysqli('localhost', ...)` or `new PDO('mysql:host=localhost;...')`. PHP opens a TCP connection to `127.0.0.1:3306` (or, on Debian/Ubuntu, the unix socket `/var/run/mysqld/mysqld.sock`).
7.  MySQL authenticates the user, parses the SQL, hits the InnoDB buffer pool (or disk if cold), returns rows.
8.  PHP renders the rows into HTML, returns it to Apache, which writes it on the wire.

The classic failure modes are annotated under the figure. The most common are:

-   **PHP shows as plain text.** Apache is serving the file but is not invoking PHP. The handler module is not loaded.
-   **Blank page after install.** PHP errors are being suppressed and the script crashed -- look in `/var/log/apache2/error.log`, not in the browser.
-   **"Connection refused" intermittently.** The MySQL connection limit is hit, or the OOM killer just shot `mysqld`. Check `dmesg` and `mysql.err`.

# 6. Installing the stack on Ubuntu

Step zero before installing anything: make sure no other web server or database is already on the box.

```bash
sudo systemctl status apache2 nginx mysql mariadb 2>/dev/null \
  | grep -E 'Active|service'
# Stop and mask anything you do not want
sudo systemctl disable --now nginx
```

The order matters: install Apache, then MySQL, then PHP last. PHP's package will pull in the Apache module and will run a post-install hook that enables it -- this only works if Apache is already there.

## 6.1 Apache

```bash
sudo apt update
sudo apt install -y apache2
sudo systemctl enable --now apache2
```

Visit `http://YOUR_PUBLIC_IP/` -- you should see the Apache2 Ubuntu Default Page. If you do not, run the four-step verification from section 4.4 in order.

The directories you will actually edit:

| Path | What lives there |
| --- | --- |
| `/etc/apache2/apache2.conf` | global config -- almost never edit directly |
| `/etc/apache2/sites-available/*.conf` | virtual host definitions |
| `/etc/apache2/sites-enabled/` | symlinks; `a2ensite` / `a2dissite` manage them |
| `/etc/apache2/mods-available/*.{load,conf}` | module config -- managed by `a2enmod` |
| `/var/www/html/` | default DocumentRoot |
| `/var/log/apache2/{access,error}.log` | the first place to look when anything fails |

A small but worth-it tweak: increase logging detail temporarily during setup, then revert.

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

The `mysql_secure_installation` wizard asks five questions. The right answers are:

1.  **VALIDATE PASSWORD plugin:** yes, level 2 (strong).
2.  **Set root password:** a 16+ character password from a password manager. Save it.
3.  **Remove anonymous users:** yes.
4.  **Disallow root login remotely:** yes -- you will tunnel via SSH.
5.  **Remove test database:** yes.

Then verify:

```bash
sudo systemctl status mysql
sudo mysql -e "SELECT version(), @@hostname;"
```

### The `caching_sha2_password` trap

MySQL 8.0 changed the default authentication plugin from `mysql_native_password` to `caching_sha2_password`. Older PHP `mysqli` drivers, the `mysql` PHP extension, and a number of CMS installers cannot speak the new protocol and fail with `The server requested authentication method unknown to the client`. The right fix today is to upgrade the driver; the pragmatic fix when you cannot is to tell MySQL to use the old plugin **per user**:

```sql
ALTER USER 'discuz_user'@'localhost'
  IDENTIFIED WITH mysql_native_password BY 'StrongPassword123!';
FLUSH PRIVILEGES;
```

Do this for the application user only -- never weaken `root`.

### A starter `my.cnf` worth knowing about

Out of the box MySQL 8 ships with a tiny buffer pool. For a site that is even slightly busy this is the biggest single performance lever:

```ini
# /etc/mysql/mysql.conf.d/zz-tuning.cnf
[mysqld]
innodb_buffer_pool_size = 4G          # ~50% of system RAM on a DB-only box
innodb_log_file_size    = 512M
innodb_flush_log_at_trx_commit = 1    # 2 if you can tolerate 1s data loss for speed
max_connections = 200
character-set-server   = utf8mb4
collation-server       = utf8mb4_unicode_ci
```

Restart MySQL after editing. The buffer pool is single-handedly responsible for the difference between "every query hits disk" and "the working set lives in RAM".

## 6.3 PHP

```bash
sudo apt install -y php libapache2-mod-php php-mysql \
                    php-curl php-gd php-mbstring php-xml php-zip php-intl
sudo systemctl restart apache2
```

Verify the bridge between Apache and PHP:

```bash
echo '<?php phpinfo();' | sudo tee /var/www/html/info.php
curl -s http://127.0.0.1/info.php | head -n 5
```

You should see HTML, not PHP source. If you see source, the `php` module did not get enabled:

```bash
ls /etc/apache2/mods-enabled/php*.load   # is there one?
sudo a2enmod php8.1                      # match your installed version
sudo systemctl restart apache2
```

**Delete `info.php` after testing** -- it leaks the entire PHP configuration, including loaded extensions, file paths and `disable_functions`. It is the first thing an attacker grep s for.

```bash
sudo rm /var/www/html/info.php
```

# 7. Defence in depth: hardening the public surface

![Defence in depth for a public LAMP server](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lamp-on-ecs/fig4_security_setup.png)

A public LAMP server with default settings will be probed by automated scanners within minutes. Treat security as five concentric rings, each one buying time even when the one outside it fails.

## 7.1 Security group -- the perimeter

Already covered in section 4. The rule of thumb: your security group should make the OS firewall feel redundant, and your OS firewall should make the security group feel redundant. Neither should be your only line.

## 7.2 OS hardening

```bash
# Keep the system patched -- enable unattended security upgrades
sudo apt install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

# Disable password SSH (after you have key auth working)
sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# Install fail2ban to slow brute force
sudo apt install -y fail2ban
sudo systemctl enable --now fail2ban
```

## 7.3 TLS with Let's Encrypt

Once you have a domain pointed at your IP, getting a certificate is two commands:

```bash
sudo apt install -y certbot python3-certbot-apache
sudo certbot --apache -d example.com -d www.example.com \
             --redirect --agree-tos -m you@example.com
```

`certbot --apache` writes a new vhost on port 443, enables `mod_ssl` and `mod_rewrite`, and adds a 301 redirect from 80 to 443. It also drops a systemd timer that renews the cert before expiry; verify with:

```bash
sudo systemctl list-timers | grep certbot
sudo certbot renew --dry-run
```

A modern TLS config does not just turn TLS on; it turns the bad parts off. After certbot, edit `/etc/apache2/sites-available/example.com-le-ssl.conf` and add:

```apache
SSLProtocol             all -SSLv3 -TLSv1 -TLSv1.1
SSLCipherSuite          HIGH:!aNULL:!MD5:!3DES
SSLHonorCipherOrder     on
Header always set Strict-Transport-Security "max-age=63072000"
```

## 7.4 MySQL hardening

-   Bind to `127.0.0.1` only (default in modern packages, verify in `/etc/mysql/mysql.conf.d/mysqld.cnf`).
-   One database user **per application**, with `GRANT` scoped to that database.
-   No `GRANT ALL ... TO root@'%'` -- ever.
-   Backups encrypted at rest if the data is sensitive.

## 7.5 Application hygiene

-   `php-fpm` instead of `mod_php` if you can -- isolates PHP failures from the Apache process tree.
-   `expose_php = Off` and `display_errors = Off` in `/etc/php/8.1/apache2/php.ini` for production.
-   Whatever framework you deploy, check it has a security advisory feed and subscribe to it. CVEs in CMSes are the single largest source of compromised LAMP servers.

# 8. End-to-end deployment: Discuz!

Discuz! is worth using as a worked example because it exercises every weak point of a fresh LAMP install: file permissions, multiple writable directories, MySQL user creation, PHP extension requirements and a web-based installer that double-checks all of them.

## 8.1 Download

```bash
cd /var/www/html
sudo wget https://download.comsenz.com/DiscuzX/3.4/Discuz_X3.4_SC_UTF8.zip
sudo apt install -y unzip
sudo unzip -q Discuz_X3.4_SC_UTF8.zip
sudo mv upload/* upload/.htaccess . 2>/dev/null || sudo mv upload/* .
sudo rm -rf upload Discuz_X3.4_SC_UTF8.zip readme.txt utility/
```

## 8.2 Permissions -- the part everyone gets wrong

Apache runs as `www-data` (Ubuntu) or `apache` (CentOS). The single rule: **the user running Apache must own every file that PHP needs to write**, and only those.

```bash
# Baseline: everything readable, owned by web user
sudo chown -R www-data:www-data /var/www/html
sudo find /var/www/html -type d -exec chmod 755 {} \;
sudo find /var/www/html -type f -exec chmod 644 {} \;

# Discuz's writable directories (the installer checks these)
for d in data config uc_server/data uc_client/data; do
  sudo chmod -R 775 /var/www/html/$d
done
```

Note that this is `775`, **not** `777`. If `www-data` already owns the directory, `775` lets the owner (web user) write while keeping `o+r` for the rest. `chmod 777` is folk wisdom, not advice -- it lets every user on the system write your application files, and on a shared server that is a privilege-escalation path.

## 8.3 Database account

```bash
sudo mysql -e "
  CREATE DATABASE discuz CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
  CREATE USER 'discuz_user'@'localhost' IDENTIFIED BY 'StrongPassword123!';
  GRANT ALL PRIVILEGES ON discuz.* TO 'discuz_user'@'localhost';
  FLUSH PRIVILEGES;
"
```

Two things to notice:

-   `discuz.*` -- the grant is scoped to one database. If Discuz is ever compromised, the attacker cannot read your other applications' tables.
-   `'discuz_user'@'localhost'` -- the host part is part of the identity. The same username from a different host is a different user. Connections via the unix socket count as `'localhost'`; TCP to `127.0.0.1` counts as `'127.0.0.1'`. If `mysql_secure_installation` left `localhost` and `127.0.0.1` distinct, grant both.

## 8.4 Run the installer

Visit `http://YOUR_PUBLIC_IP/install/`. Three things happen:

1.  **Environment check** -- PHP version, GD, mbstring, mysqli. If anything is missing: `sudo apt install -y php-<extension> && sudo systemctl reload apache2`.
2.  **Permission check** -- the green ticks should appear next to `data/`, `config/`, `uc_server/data/`, `uc_client/data/`. If not, recheck section 8.2.
3.  **Database details** -- host `localhost`, name `discuz`, user `discuz_user`, password as set above.

After install:

```bash
sudo rm -rf /var/www/html/install
# tighten the directories that no longer need write access
sudo chmod -R 755 /var/www/html/config
```

# 9. The five failures that hit everyone

## Failure 1 -- "Connection refused"

**Means:** something between you and Apache is dropping the TCP SYN, or Apache is not listening.

```bash
# Local on the server -- if this works, Apache is fine; the problem is the network
curl -I http://127.0.0.1

# What is actually listening?
sudo ss -tlnp | grep -E ':80|:443'

# What does the OS firewall say?
sudo ufw status
```

If `127.0.0.1` works but the public IP does not, the OS is fine -- check the security group in the cloud console.

## Failure 2 -- "403 Forbidden" or "Index of /"

**Means:** Apache served the directory but did not find an index file, or could not read it.

```bash
ls -l /var/www/html/                       # is index.php / index.html there?
sudo -u www-data cat /var/www/html/index.php # can the web user read it?
grep -r DirectoryIndex /etc/apache2/
```

The fix is almost always `chown -R www-data:www-data /var/www/html` -- you `wget` ed something as root, and the web user cannot read it.

## Failure 3 -- PHP source code visible in the browser

**Means:** Apache is serving `.php` as a static file because the PHP handler is not registered.

```bash
apache2ctl -M | grep php_module    # nothing? PHP module is off
sudo a2enmod php8.1                # use your version
sudo systemctl restart apache2
curl -s http://127.0.0.1/info.php | head -n 1   # should be HTML, not <?php
```

This is a security incident, not just a misconfiguration -- never leave the box exposed in this state.

## Failure 4 -- "Can't connect to MySQL server on 'localhost'"

**Means:** MySQL is down, the socket has moved, or credentials are wrong.

```bash
sudo systemctl status mysql
sudo journalctl -u mysql -n 100 --no-pager
mysql -u discuz_user -p -h 127.0.0.1 -e 'SELECT 1'
```

A common cause on small instances: MySQL was OOM-killed. `dmesg | tail -50` will show `Killed process ... mysqld`. Either tune `innodb_buffer_pool_size` down or move to a larger instance.

## Failure 5 -- Discuz says "Directory not writable"

**Means:** the web user cannot write to one of the four required dirs.

```bash
sudo -u www-data touch /var/www/html/data/.write_test
# Permission denied? Then:
sudo chown -R www-data:www-data /var/www/html/{data,config,uc_server/data,uc_client/data}
sudo chmod -R 775 /var/www/html/{data,config,uc_server/data,uc_client/data}
```

Resist the urge to `chmod -R 777 /var/www`. It will work, and it will hurt later.

# 10. Production essentials

## 10.1 Virtual hosts

Stop dumping everything in `/var/www/html/` the moment you have more than one site. Per-site directories under `/var/www/<sitename>/` and per-site vhost files keep the layout sane.

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

`configtest` before `reload` is the difference between a graceful change and a five-minute outage when you mistype a brace.

## 10.2 Backups that you actually test

A backup you have not restored is not a backup. The minimum:

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
# keep 14 days
find "$DEST" -name 'discuz_*.sql.gz' -mtime +14 -delete
```

```cron
# crontab -e
0 3 * * * /usr/local/bin/db-backup.sh
```

Add an OSS sync once a week so a failed disk does not also lose your backups:

```bash
ossutil cp -r /var/backups/mysql/ oss://mybucket/db-backups/$(hostname)/
```

And once a month, on a separate machine: `gunzip < some_backup.sql.gz | mysql -u root -p test_restore` and verify the row counts. The first time is always educational.

## 10.3 Observability

The Aliyun Cloud Monitor agent gives you CPU, memory, disk and bandwidth out of the box. The two extra signals worth wiring up yourself:

-   Apache `mod_status` exposed on `127.0.0.1:80/server-status` -- requests per second, busy workers, slow requests.
-   MySQL `performance_schema` queries to find the slow queries (`SELECT digest_text, count_star, avg_timer_wait FROM events_statements_summary_by_digest ORDER BY sum_timer_wait DESC LIMIT 10`).

A weekly five-minute look at these will catch capacity problems weeks before they bite.

# 11. Two topologies, one app

![Two topologies for a LAMP site](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/lamp-on-ecs/fig5_deployment_topology.png)

Almost every successful LAMP site eventually hits the wall of the single-instance topology and has to decide whether to scale up (bigger ECS) or out (split tiers). The picture above is the destination of that decision.

**Stay all-in-one when** your peak traffic fits in one instance, your DB is small enough that the buffer pool covers the working set, and you have one engineer. The localhost connection between PHP and MySQL is faster than any network call you can buy, and the operational footprint is one OS to patch.

**Split into three tiers when** you need horizontal scale (more PHP workers behind an SLB), high availability (RDS gives you primary + standby for free), or you are spending more on your single ECS than two smaller ones plus an RDS. The classic Aliyun three-tier:

-   **SLB** terminates TLS, fans out to the web tier.
-   **ECS x N** running Apache + PHP, all stateless (sessions in Redis, uploads on OSS).
-   **RDS for MySQL** as the single source of truth.

The cost roughly triples; the failure surface goes from "one box" to "many boxes plus a network", which is genuinely harder to operate. Do not migrate just because the diagrams look impressive -- migrate because the single instance is actually saturating.

# 12. Compiling MySQL from source (advanced)

You normally do not need to do this. Use the package manager unless you have a concrete reason -- a build flag the package omits, a pinned version your vendor mandates, a patch the upstream has not merged. The downsides of source builds are real: hours of compile time, no automatic security updates, your own job to track CVEs.

If you do need it, the canonical incantation for MySQL 5.6 on CentOS:

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

sudo make -j$(nproc)         # 1-3 hours on a 2 vCPU box
sudo make install

sudo useradd -r -s /sbin/nologin mysql
sudo chown -R mysql:mysql /usr/local/mysql
cd /usr/local/mysql
sudo ./scripts/mysql_install_db --user=mysql

sudo cp support-files/mysql.server /etc/init.d/mysql
sudo systemctl enable --now mysql
```

Two things go wrong almost every time:

1.  **`Could not find OpenSSL`** -- you missed `openssl-devel`. Fix: install it, then **remove the build directory and re-extract** before retrying. `cmake` caches partial state and a half-finished tree will not pick up the new headers.
2.  **OOM during compilation** -- `make -j$(nproc)` on a 2 GiB instance will be killed. Use `-j2` and add 2 GiB of swap before starting.

After install, do not forget the same `mysql_secure_installation` and `my.cnf` tuning from section 6.2 -- a from-source build is not configured for you.

# 13. Real-world cases

## Case A -- Migrating WordPress from shared hosting

The recipe that has worked dozens of times:

```bash
# On the old host (cPanel / phpMyAdmin / SSH)
mysqldump --single-transaction wordpress > wp.sql
tar czf wp-files.tar.gz public_html/

# Move both to the ECS via scp / OSS
scp wp.sql wp-files.tar.gz user@8.134.207.88:/tmp/

# On the ECS
cd /var/www/html
sudo tar xzf /tmp/wp-files.tar.gz --strip-components=1
sudo mysql -e "CREATE DATABASE wordpress; \
               CREATE USER 'wp'@'localhost' IDENTIFIED BY '...'; \
               GRANT ALL ON wordpress.* TO 'wp'@'localhost';"
sudo mysql wordpress < /tmp/wp.sql

# Update wp-config.php with the new DB credentials
# Update the site URL in two places:
sudo mysql wordpress -e "UPDATE wp_options SET option_value='https://example.com' \
                         WHERE option_name IN ('siteurl','home');"

# Permissions
sudo chown -R www-data:www-data /var/www/html
sudo find /var/www/html -type d -exec chmod 755 {} \;
sudo find /var/www/html -type f -exec chmod 644 {} \;

# DNS A record -> 8.134.207.88, TTL low for cutover
```

The pitfall is almost always permissions -- shared hosts give you ownership of everything; on ECS, `www-data` does, and uploads will silently fail until you fix it.

## Case B -- Two PHP versions on one box

Old plugin needs 5.6, new app wants 8.1. Use `php-fpm` per version and route by vhost:

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

This is also the right time to leave `mod_php` behind. `php-fpm` runs PHP in its own process pool, with its own user, its own resource limits, and its own crash recovery -- a memory leak in PHP no longer takes Apache down.

## Case C -- A site that suddenly returns 502s under load

A common pattern: traffic doubles, the site starts returning 502 to about 5% of requests. The chain of cause is almost always:

1.  Apache prefork hits `MaxRequestWorkers`; new connections queue.
2.  PHP-FPM hits `pm.max_children`; Apache gets a 502 from the FPM socket.
3.  MySQL hits `max_connections`; PHP-FPM workers block waiting for a connection, then time out.

The fix is to size each tier to one above the next. A starting point on a 4 vCPU / 16 GiB instance:

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

The numbers are not magic; the principle is. Each layer's worker pool must be able to absorb the layer above it without queueing for too long.

# 14. Summary

LAMP on Aliyun ECS reduces to a five-step recipe:

1.  **Open ports correctly** -- security group, then OS firewall, then verify hop by hop.
2.  **Install in order** -- Apache, MySQL, PHP, in that order, each one verified before moving on.
3.  **Verify each layer** -- Apache serves HTML, PHP runs, MySQL connects. Three commands, every time.
4.  **Set permissions deliberately** -- `www-data` owns the writable parts, no `chmod 777`.
5.  **Deploy the app** -- whether Discuz, WordPress or your own PHP, the playbook is the same.

What to do next:

-   Add HTTPS with Let's Encrypt and set HSTS.
-   Wire up `mysqldump -> OSS` and **restore from backup** at least once.
-   Read your access log for an hour -- you will learn more about your traffic and your attackers than from any blog post.
-   Once you outgrow one box, split into SLB + ECS + RDS rather than scaling the single instance forever.

Further reading:

-   Apache HTTP Server documentation -- <https://httpd.apache.org/docs/>
-   MySQL 8.0 reference manual -- <https://dev.mysql.com/doc/refman/8.0/en/>
-   PHP manual -- <https://www.php.net/manual/en/>
-   Aliyun ECS user guide -- <https://www.alibabacloud.com/help/en/ecs/>
-   Let's Encrypt with Certbot -- <https://certbot.eff.org/instructions>
