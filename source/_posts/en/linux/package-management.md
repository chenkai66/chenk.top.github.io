---
title: "Linux Package Management: apt, dnf, pacman, and Building from Source"
date: 2024-01-24 09:00:00
tags:
  - Linux
  - Cloud
  - System Administration
categories: Linux
series: "Linux"
series_order: 4
series_total: 8
lang: en
mathjax: false
description: "Master package management across distributions: dpkg/apt for Debian/Ubuntu, rpm/yum/dnf for RHEL/CentOS, pacman for Arch, dependency troubleshooting, version locking, mirrors, and compiling from source."
---

Most people learn package management as three commands: `install`, `remove`, `upgrade`. That works until something goes wrong - a dependency conflict, an upgrade that won't apply, a kernel that doesn't boot, a mirror that times out from inside China. At that point you need a model of what is actually happening: what a *package* contains, what the *manager* is solving for, where it stores state, and how the difference between Debian's `apt/dpkg` and Red Hat's `dnf/rpm` shows up at 2 a.m. on a production box.

This article is the model plus the cookbook. We start with what is in a `.deb` or `.rpm` file and why a manager is needed at all. Then we walk through `apt`, `dnf` and `pacman` side by side - not just "the equivalent commands" but where they share assumptions and where they diverge (dependency resolution, version pinning, downgrade, repo trust). After that we cover the things you only learn the hard way: configuring a domestic mirror that actually works, building Nginx from source when the repo version is too old, dropping in a binary tarball like the JDK without `apt`, and a small set of habits that keep production boxes upgradable for years.

![Mainstream Linux package toolchains](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/package-management/fig1_package_manager_comparison.png)

# What a package actually is

A package is a single file (`.deb`, `.rpm`, `.pkg.tar.zst`) that contains everything one piece of software needs *plus* metadata describing how it should be installed. The metadata is what makes a package a package - without it you have a tarball.

Concretely, a typical server package contains six kinds of payload:

**1. Executables.** Compiled binaries that go under `/usr/bin` (regular user commands) or `/usr/sbin` (system administration). Examples: `/usr/bin/vim`, `/usr/sbin/nginx`, `/usr/bin/gcc`.

**2. Configuration files.** The defaults that ship with the software, almost always under `/etc`. The package manager treats these specially: on upgrade it will *not* clobber a config file you have edited. Apt prompts you (`Y/I/N/O/D/Z`) and keeps a `.dpkg-dist` next to your file; dnf writes the new defaults as `.rpmnew` and leaves yours alone. This single rule has saved countless production outages.

```text
/etc/nginx/nginx.conf       # web server
/etc/mysql/my.cnf           # database
/etc/ssh/sshd_config        # SSH daemon
```

**3. Shared libraries.** The `.so` files that executables link against at runtime - the Linux equivalent of Windows DLLs. They sit under `/usr/lib`, `/lib`, or distro-specific paths like `/lib/x86_64-linux-gnu/`. Sharing them saves disk and memory (one copy serves every consumer) and means a single security upgrade to `libssl3` patches every TLS client on the box.

```text
/usr/lib/x86_64-linux-gnu/libssl.so.3   # OpenSSL
/lib/x86_64-linux-gnu/libc.so.6         # GNU C library
```

**4. Data files.** Anything the program needs at runtime that is not code or config: the empty database template under `/var/lib/mysql`, the default web root under `/var/www/html`, locale data, sample certificates, etc.

**5. Documentation.** Manual pages under `/usr/share/man/`, README and changelog under `/usr/share/doc/<pkg>/`. Often the first place to look before reaching for a search engine.

**6. Service unit files.** If the package ships a daemon, it drops a `.service` file under `/usr/lib/systemd/system/` so that `systemctl enable nginx` works the moment the install finishes. These are owned by the package; admin overrides go under `/etc/systemd/system/` to avoid being overwritten on upgrade.

On top of those payloads, the package metadata records: the package name and version, dependencies (`Depends:` in `.deb`, `Requires:` in `.rpm`), conflicts, post-install scripts (e.g. `useradd nginx`), and a SHA256 of every file. That last bit is what lets `dpkg -V` or `rpm -V` later tell you which files have been tampered with.

## Why we need a package manager

Without one you would, for every piece of software:

1. Find a download link (and trust whoever runs the site).
2. Manually resolve dependencies - A needs B 1.1+, B needs C, C conflicts with D you already installed.
3. Decide where each file goes.
4. Remember every file you copied so you can uninstall it later.
5. Poll upstream by hand for security updates.

A package manager replaces all of that with a database (`/var/lib/dpkg/` or the `rpmdb`) plus a solver. The database knows every file every installed package owns; the solver knows every package available in the configured repositories and can plan an install, upgrade or removal that is consistent. That is the entire value proposition - and it is enormous.

# The major package toolchains

The Linux ecosystem split into a handful of package families decades ago and has been remarkably stable since. The names you actually need to know:

| Family       | Distros                              | Format          | Low-level | High-level         |
|--------------|--------------------------------------|-----------------|-----------|--------------------|
| Debian       | Debian / Ubuntu / Mint               | `.deb`          | `dpkg`    | `apt`, `apt-get`   |
| Red Hat      | RHEL / CentOS / Rocky / Alma / Fedora| `.rpm`          | `rpm`     | `dnf` (`yum` on EL7) |
| Arch         | Arch / Manjaro / EndeavourOS         | `.pkg.tar.zst`  | `pacman`  | `pacman`, `yay`    |
| SUSE         | openSUSE / SLES                      | `.rpm`          | `rpm`     | `zypper`           |
| Gentoo       | Gentoo                               | source `ebuild` | -         | `portage` (`emerge`)|

The split that matters in practice is **low-level vs high-level**. Low-level tools (`dpkg`, `rpm`, plain `pacman -U`) operate on a single file: they unpack, run scripts and update the local database, but they will *not* fetch missing dependencies. High-level tools (`apt`, `dnf`, `pacman -S`, `zypper`) wrap the same actions but talk to remote repositories and run a dependency solver. Most of the time you want the high-level tool. The low-level tool is what you reach for when something the high-level tool would not let you do is exactly what you need - like force-installing a package that conflicts on paper, or extracting a `.deb` to inspect its contents.

The rest of the article focuses on `apt` (Debian/Ubuntu), `dnf` (RHEL/Rocky/Fedora) and `pacman` (Arch), with notes on the underlying `dpkg` / `rpm` tools where they matter.

# Dependency resolution: the actual hard part

![Dependency resolution graph](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/package-management/fig2_dependency_resolution.png)

The most surprising thing about package managers is that the bulk of their code is not the install logic - it is the solver. Here is what `apt install nginx` actually has to do:

1. **Read the metadata** for `nginx` from the cached `Packages` index.
2. **Walk the dependency graph** - `nginx` depends on `libssl3`, `libpcre2-8`, `zlib1g`, `libc6`, ...; each of those depends on more things; you stop when you hit something already installed at an acceptable version.
3. **Unify versions.** If two paths through the graph want different versions of `libc6`, the solver has to either find a single version that satisfies both, or report a conflict. There is no "install both" option for shared libraries - the dynamic loader resolves names, not paths, and you get one binding.
4. **Choose an install order** that respects "this must be installed before that" so post-install scripts don't run with their dependencies missing.
5. **Fail loudly** if no consistent solution exists, rather than half-installing and leaving the system in a bent state.

When you skip the high-level tool, you lose all of this:

```bash
sudo dpkg -i nginx_1.24.0-1_amd64.deb
# dpkg: dependency problems prevent configuration of nginx:
#  nginx depends on libssl3 (>= 3.0.0); however:
#   Package libssl3 is not installed.
sudo apt install -f         # ask apt to fix what dpkg broke
```

The `apt install -f` step is the dependency solver running *after the fact* to repair a partial state. You can avoid the whole dance by letting the high-level tool handle the local file in the first place:

```bash
sudo apt install ./nginx_1.24.0-1_amd64.deb     # apt resolves deps from configured repos
sudo dnf install ./nginx-1.24.0-1.x86_64.rpm    # same on Red Hat family
```

The same logic applies to *removal*. If you have `nginx` and `libcache-extra` installed, removing only `libcache-extra` is fine - but uninstalling `nginx` should also remove the now-orphaned `libpcre2-8` if nothing else needs it. That is what `apt autoremove` and `dnf autoremove` do.

# Debian / Ubuntu: apt and dpkg

## When to use which

| Tool       | What it is                                                                    | When to use                              |
|------------|-------------------------------------------------------------------------------|------------------------------------------|
| `apt`      | Modern user-facing front-end. Colour, progress bars, sane defaults.           | Interactive shells, day to day.          |
| `apt-get`  | Older front-end with the same engine. Stable, parseable output.               | Shell scripts, Ansible, Dockerfiles.     |
| `dpkg`     | The low-level tool. Operates on one `.deb` at a time. No dependency solver.   | Inspect a package, force a state, rescue.|

A useful mental model: `apt` is `apt-get` plus interactive niceties; `apt-get` is the dependency solver wrapping `dpkg`; `dpkg` is what actually unpacks files and runs maintainer scripts.

## Daily commands

```bash
sudo apt update                       # refresh repository indexes (no installs)
sudo apt upgrade                      # apply available upgrades, never remove anything
sudo apt full-upgrade                 # apply upgrades, may remove packages if needed
sudo apt install nginx                # install with deps
sudo apt install nginx=1.24.0-2ubuntu1   # pin a specific version at install time
sudo apt remove nginx                 # uninstall but keep /etc/nginx
sudo apt purge nginx                  # uninstall and wipe config
sudo apt autoremove                   # drop dependencies nothing needs anymore
```

Two commands new admins reliably get wrong:

- `apt update` does not install anything. It downloads the latest `Packages` indexes from your mirrors so the next `apt install` knows what is available. Forgetting to run it is the reason "the package exists, but I can't install it."
- `apt remove` keeps `/etc`. That is usually what you want (reinstalling later restores your config), but if you are decommissioning the service for good, `apt purge` is the right call.

## Searching, inspecting, locating

```bash
apt search nginx                  # search the configured repos
apt show nginx                    # version, deps, description, size
apt-cache policy nginx            # candidate version + which repo it would come from
dpkg -l                           # list everything installed
dpkg -l | grep nginx              # is nginx installed?
dpkg -L nginx                     # list every file nginx installed
dpkg -S /usr/sbin/nginx           # which package owns this file?
```

The last two are operational lifesavers. "What did this package put on my disk?" and "Which package does this random binary belong to?" come up constantly during incident response, and there is no Linux equivalent of running `dpkg -S` against an unknown file in the path.

## Pinning, downgrading, locking

```bash
sudo apt-mark hold nginx          # never auto-upgrade nginx
apt-mark showhold                 # what is currently held
sudo apt-mark unhold nginx        # release the hold

# Downgrade in three steps
apt-cache policy nginx            # 1) see candidate versions
sudo apt install nginx=1.22.1-1ubuntu1   # 2) install the older one
sudo apt-mark hold nginx          # 3) freeze it so the next upgrade won't overwrite
```

Holds are how you survive a known-bad upstream release: pin the previous version, schedule the proper upgrade for a maintenance window, move on.

## Cleaning up

```bash
sudo apt clean        # delete every cached .deb under /var/cache/apt/archives
sudo apt autoclean    # only delete cached .debs no longer in any repo
sudo apt autoremove   # remove orphan dependencies
```

A single combined cleanup that I keep in muscle memory:

```bash
sudo apt update && sudo apt upgrade -y && sudo apt autoremove -y && sudo apt autoclean
```

Run it once a week on every box you own and your `/var` partition will thank you.

# RHEL / Rocky / Fedora: dnf (and yum, and rpm)

`dnf` replaced `yum` in CentOS 8 / RHEL 8 and Fedora 22. The command surface is almost identical (a deliberate design goal), but the engine underneath is rewritten in Python with a proper SAT solver (`libsolv`), which is why `dnf` operations feel noticeably faster on large transactions. CentOS 7 still ships `yum`; everywhere else assume `dnf`.

## Daily commands

```bash
sudo dnf makecache                 # refresh metadata
sudo dnf check-update              # what could be upgraded?
sudo dnf upgrade                   # apply all upgrades
sudo dnf install nginx             # install
sudo dnf install nginx-1.24.0      # specific version
sudo dnf remove nginx              # uninstall
sudo dnf autoremove                # drop orphans
sudo dnf downgrade nginx-1.22.1    # native downgrade (apt has no equivalent verb)
```

`dnf downgrade` is one of the things `dnf` does better than `apt` - in the apt world a downgrade is "install the older version and hope it doesn't conflict"; in the dnf world it is a first-class operation.

## Searching, inspecting, locating

```bash
dnf search nginx                   # search repos
dnf info nginx                     # detailed info
dnf list --showduplicates nginx    # every version available across repos
rpm -qa                            # all installed packages
rpm -qa | grep nginx               # is it installed?
rpm -ql nginx                      # files installed by nginx
rpm -qf /usr/sbin/nginx            # which package owns this file?
rpm -qc nginx                      # only the config files
rpm -V nginx                       # verify nothing has been tampered with
```

`rpm -V` is the underrated one. Output like `S.5....T.  c /etc/nginx/nginx.conf` decodes as: size differs (`S`), MD5 differs (`5`), mtime differs (`T`), and it is a config file (`c`). Config files are expected to differ from the original; binaries differing under `/usr/sbin/` are how you find that something tampered with the system.

## Pinning and history

```bash
sudo dnf install 'dnf-command(versionlock)'   # one-time: install the plugin
sudo dnf versionlock add nginx                # pin nginx
dnf versionlock list                          # what is pinned
sudo dnf versionlock delete nginx             # release

dnf history                                   # every transaction this box has ever done
dnf history info 42                           # what exactly did transaction 42 install/remove?
sudo dnf history undo 42                      # roll it back
```

`dnf history` has no real `apt` equivalent. It records every transaction with timestamps, command line, and full diff, and it can roll any of them back. On servers that have lived through several admins this is how you reconstruct what happened.

# Arch: pacman

Arch Linux uses `pacman` for both low-level and high-level work. It is rolling-release - there is no "Arch 22.04," just "what the repos look like right now" - so the workflow is biased toward "always upgrade everything together":

```bash
sudo pacman -Syu                  # sync repos AND upgrade everything (the canonical command)
sudo pacman -S nginx              # install
sudo pacman -Ss nginx             # search
sudo pacman -Si nginx             # info
sudo pacman -Qi nginx             # info on the installed package
sudo pacman -Ql nginx             # files owned by nginx
sudo pacman -Qo /usr/bin/nginx    # which package owns this file?
sudo pacman -R nginx              # remove (keeps deps)
sudo pacman -Rns nginx            # remove + orphan deps + config files
sudo pacman -Sc                   # clean cache
```

Arch's golden rule: do not partial-upgrade. `pacman -S nginx` after weeks of not running `pacman -Syu` can pull in a `nginx` built against newer libraries than you have, and you end up with a broken `nginx`. Always sync the system before installing anything new, or just use `pacman -Syu nginx` to do both atomically.

The Arch User Repository (AUR) - community-maintained build recipes - is fronted by tools like `yay` or `paru`, which wrap `pacman` plus `makepkg` to fetch source, build, and install in one step. Useful, but treat anything from the AUR as "I will read this PKGBUILD before installing it."

# The lifecycle on one box

![Package lifecycle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/package-management/fig3_package_lifecycle.png)

Stepping back from the individual tools, every package on your system moves through the same lifecycle: `search -> install -> upgrade (or hold) -> remove`. The package manager records the current state of every package in its database (`/var/lib/dpkg/status` for Debian, the `rpmdb` under `/var/lib/rpm/` for Red Hat). Every command you run is, mechanically, a transaction against that database plus actions on disk to make reality match.

This is why "I deleted the binary by hand" is always wrong. The database still says the package is installed; the next upgrade of that package will silently put the file back; `dpkg -V` / `rpm -V` will report the file as "missing." Either remove the package properly (`apt purge`, `dnf remove`) or, if you really need to keep the package metadata but blank the binary, divert it (`dpkg-divert`).

# How a repository is laid out

![Repository structure and chain of trust](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/package-management/fig4_repository_structure.png)

When you run `apt update`, your machine talks HTTPS to a mirror and downloads index files in a very specific layout. Understanding that layout makes broken-mirror diagnostics a matter of `curl`-ing the right URL.

A Debian-style apt repo serves two trees:

- **`dists/<suite>/`** - per-suite metadata. The top file is `Release` (or its inline-signed cousin `InRelease`), which lists every component (`main`, `universe`, ...) and architecture, with SHA256 sums of the corresponding `Packages.gz` index files. `Release.gpg` is a detached GPG signature. This is the root of trust.
- **`pool/`** - the actual `.deb` files, organised by source-package name (`pool/main/n/nginx/nginx_1.24.0-1_amd64.deb`).

The chain of trust on `apt install nginx` is:

1. Your machine has GPG public keys it trusts under `/etc/apt/trusted.gpg.d/` (placed there by `signed-by=` in the source list, or historically by `apt-key`).
2. `apt update` fetches `InRelease`, verifies the signature against those keys, and refuses to use the suite if verification fails (`NO_PUBKEY` errors come from this step).
3. From `InRelease` it knows the SHA256 of `Packages.gz` for each component; it fetches them and verifies.
4. From `Packages.gz` it knows the SHA256 of `nginx_1.24.0-1_amd64.deb`; it fetches it from `pool/`, verifies, and only then hands it to `dpkg`.

If anything in this chain breaks - missing key, modified `Release`, mismatched checksum on the `.deb` - apt aborts. That is the security guarantee: a malicious mirror can serve you whatever bytes it wants, but it cannot get them past the chain unless it also has the upstream signing key.

The Red Hat side is structurally similar: `repodata/repomd.xml` is the signed manifest, with checksums of the `primary.xml.gz` index and the `.rpm` files under the repo. `dnf` enforces the same chain unless you explicitly disable `gpgcheck` (you should not).

## Configuring a domestic mirror (China)

The default Ubuntu and CentOS mirrors are slow from inside China. The fix is to point the source list at a domestic mirror. The two big ones are Aliyun (`mirrors.aliyun.com`) and Tsinghua (`mirrors.tuna.tsinghua.edu.cn`). Both serve every popular distro.

**Ubuntu (apt):**

```bash
# 1. back up the originals
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak

# 2. rewrite the URLs in place (Ubuntu 22.04 'jammy')
sudo sed -i.aliyun \
    -e 's|http://.*archive.ubuntu.com|http://mirrors.aliyun.com|g' \
    -e 's|http://.*security.ubuntu.com|http://mirrors.aliyun.com|g' \
    /etc/apt/sources.list

# 3. refresh and verify
sudo apt update
grep mirrors.aliyun.com /etc/apt/sources.list   # sanity check
```

Codenames for the common Ubuntu LTS releases: `bionic` (18.04), `focal` (20.04), `jammy` (22.04), `noble` (24.04). Use the right one in the sources file.

Ubuntu 24.04 moved from `/etc/apt/sources.list` to `/etc/apt/sources.list.d/ubuntu.sources` in the deb822 format. The same find-and-replace works, just on the new file.

**CentOS / Rocky (dnf):**

```bash
sudo cp /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.bak

# CentOS 7
sudo curl -o /etc/yum.repos.d/CentOS-Base.repo \
    http://mirrors.aliyun.com/repo/Centos-7.repo

# Rocky Linux 9 (CentOS 8/Stream is EOL; use Rocky or Alma in production)
sudo sed -e 's|^mirrorlist=|#mirrorlist=|g' \
         -e 's|^#baseurl=http://dl.rockylinux.org/$contentdir|baseurl=https://mirrors.aliyun.com/rockylinux|g' \
         -i.bak /etc/yum.repos.d/rocky-*.repo

sudo dnf clean all && sudo dnf makecache
dnf repolist                                # confirm aliyun appears
```

The Tsinghua mirror exposes the same paths with `mirrors.tuna.tsinghua.edu.cn` substituted; pick whichever is faster from your network. After switching, run a real upgrade (`apt upgrade -y` / `dnf upgrade -y`) - if the mirror is misconfigured you find out immediately, not the day you actually need to install something.

# Building from source: configure / make / make install

Sometimes the repo version is too old, or you need a build option that the distro package omits (an Nginx module, an OpenSSL version, a tuning flag). Source builds are the fallback. The canonical recipe is the *autotools three-step*:

1. **`./configure`** probes your system - which compiler, which libraries, which headers - and writes a `Makefile` that reflects the answers and the options you asked for.
2. **`make`** runs the compiler against the source according to that `Makefile`.
3. **`sudo make install`** copies the built artifacts into the directories the `Makefile` recorded, typically under whatever `--prefix` you gave to `configure`.

Cmake and Meson projects use different verbs (`cmake -B build`, `cmake --build build`, `cmake --install build`) but follow the same shape: configure, build, install.

## Worked example: Tengine (Nginx fork) on Ubuntu

Tengine is Taobao's Nginx fork with extra modules. The build is identical to upstream Nginx - if you can build Tengine you can build Nginx.

```bash
# 1. build dependencies
sudo apt install -y build-essential libpcre3 libpcre3-dev libssl-dev zlib1g-dev
# Red Hat equivalent:
# sudo dnf install -y gcc make pcre-devel openssl-devel zlib-devel

# 2. fetch source
wget http://tengine.taobao.org/download/tengine-2.3.3.tar.gz
tar -zxvf tengine-2.3.3.tar.gz
cd tengine-2.3.3

# 3. configure - this is where you encode build choices
./configure \
    --prefix=/usr/local/nginx \
    --with-http_ssl_module \
    --with-http_v2_module \
    --with-http_realip_module \
    --with-http_gzip_static_module

# 4. compile - parallel across all cores
make -j"$(nproc)"

# 5. install - copies binaries, default conf and html into --prefix
sudo make install
```

What each step actually did:

- `./configure` checked that you have a C compiler and the OpenSSL/PCRE/zlib development headers. If anything was missing you got a clear error - install the missing `-dev` package and re-run. It then wrote a `Makefile` that bakes in `/usr/local/nginx` as the install path and enables the four HTTP modules you asked for.
- `make` walked the `Makefile` and produced `objs/nginx` plus a few helpers.
- `sudo make install` created `/usr/local/nginx/{sbin,conf,logs,html}` and copied the binary, default config and sample HTML into them. It did *not* create a systemd unit, register a user, or open a firewall port - that is on you.

Run it:

```bash
/usr/local/nginx/sbin/nginx                    # start
/usr/local/nginx/sbin/nginx -s reload          # reload config
/usr/local/nginx/sbin/nginx -s stop            # stop
```

To get it under systemd, drop `/etc/systemd/system/nginx.service`:

```ini
[Unit]
Description=Nginx HTTP Server (built from source)
After=network.target

[Service]
Type=forking
PIDFile=/usr/local/nginx/logs/nginx.pid
ExecStart=/usr/local/nginx/sbin/nginx
ExecReload=/usr/local/nginx/sbin/nginx -s reload
ExecStop=/usr/local/nginx/sbin/nginx -s stop
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

then:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nginx
```

The catch you sign up for the moment you build from source: there is no `apt upgrade` for it. You own the upgrade path, the security tracking, and the rebuild-when-OpenSSL-changes work. For most production workloads the right answer is "use the distro package unless I have a concrete reason not to." Source builds are for when you have that reason.

# Binary tarballs: the JDK example

Some software ships as "extract and run" binary archives - JDKs, Go, Node.js, most database engines. There is no installer to argue with; the filesystem layout and the environment variables are entirely your call.

```bash
# 1. fetch (Adoptium / Oracle / your internal mirror)
wget https://download.example.com/jdk-17_linux-x64_bin.tar.gz

# 2. extract under /opt - the convention for "third-party self-contained software"
sudo tar -zxf jdk-17_linux-x64_bin.tar.gz -C /opt
sudo mv /opt/jdk-17* /opt/jdk-17                # short, stable path

# 3. expose it system-wide
sudo tee /etc/profile.d/jdk17.sh > /dev/null <<'EOF'
export JAVA_HOME=/opt/jdk-17
export PATH=$JAVA_HOME/bin:$PATH
EOF

# 4. pick it up in the current shell
source /etc/profile.d/jdk17.sh

# 5. verify
java -version
javac -version
```

`/etc/profile.d/*.sh` is sourced by login shells for every user, which is the right scope for a JDK. For per-user installs use `~/.bashrc` or `~/.zshrc` instead. To run multiple JDK versions side by side, drop them all under `/opt/jdk-XX` and switch via `JAVA_HOME` (or use a tool like `sdkman` / `jenv`).

The same pattern works for Go (`/opt/go`), Node.js (`/opt/node-vXX`), and so on. They are popular precisely because they sidestep the distro release cycle - useful when you need a specific upstream version and the distro is six months behind.

# Modern alternatives: Snap, Flatpak, AppImage

![Snap, Flatpak, AppImage vs distro packages](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linux/package-management/fig5_modern_alternatives.png)

The distro package model has one persistent weakness: it ties a piece of software to the rest of the system. A Firefox built against Ubuntu 22.04's `glibc` and `gtk` cannot easily be shipped to RHEL 8. Three projects address this differently:

- **Snap** (Canonical) - bundles dependencies inside a `.snap` archive (a SquashFS image), runs apps under AppArmor confinement, and updates automatically via `snapd`. Default for some Ubuntu apps (Firefox, the Chromium snap). Ubiquitous on Ubuntu, less so elsewhere.
- **Flatpak** (freedesktop.org / RHEL community) - bundles apps against shared *runtimes* (e.g. `org.freedesktop.Platform//22.08`), sandboxed via `bubblewrap`, distributed mostly via Flathub. The de-facto standard for Linux desktop GUI apps.
- **AppImage** - one self-contained executable file, no install step, no daemon, no sandbox by default. Double-click to run. Great for "I just want to try this app" or for distributing internal tools.

```bash
# install
sudo apt install snapd                                  # one-time
sudo snap install firefox

sudo dnf install flatpak                                # one-time
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install flathub org.gimp.GIMP

# AppImage: literally just download and chmod +x
wget https://example.com/foo.AppImage
chmod +x foo.AppImage
./foo.AppImage
```

When to reach for which:

- **Server-side stuff** (`nginx`, `mysql`, `python`, `redis`) - distro packages, every time. They are smaller, integrate with systemd, and security updates land via the distro security team.
- **Desktop GUI apps** that ship faster than the distro - Flatpak (Flathub has the widest selection) or Snap.
- **One-off binaries** you want to evaluate without polluting the system - AppImage.

You can mix all four on the same machine. They install into separate roots, use separate update mechanisms, and do not collide.

# Habits that keep production boxes upgradable

These are the rules I have ended up with after years of fixing other people's boxes:

**1. Update on a schedule.** Not "when something breaks." Weekly `apt upgrade -y` / `dnf upgrade -y` on every server, gated through whatever change-management you have. The longer a box goes without updates, the harder the eventual upgrade gets.

**2. Clean as you go.**

```bash
sudo apt autoremove -y && sudo apt autoclean      # Debian/Ubuntu
sudo dnf autoremove -y && sudo dnf clean all      # Red Hat family
```

When `/var` fills up unexpectedly, the package cache is the first thing to check:

```bash
du -sh /var/cache/apt/archives    # Debian/Ubuntu
du -sh /var/cache/dnf             # Red Hat family
du -sh /var/lib/snapd/cache       # snap revisions add up fast
```

**3. Pin what you cannot afford to upgrade silently.** Database engines, kernels on workloads with custom modules, anything you have benchmarked at a specific version. Hold it (`apt-mark hold` / `dnf versionlock add`) and upgrade it deliberately.

**4. Do not mix package manager and source builds for the same software.** If `apt install nginx` and `make install` both put an `nginx` on your `PATH`, you will eventually start the wrong one, edit the wrong config and waste an hour figuring it out. Pick one.

**5. Source-built software lives under `/opt` or `/usr/local`.** The package manager owns `/usr` (except `/usr/local`). Stay out of its way.

**6. Back up `/etc` before touching it.**

```bash
sudo cp /etc/nginx/nginx.conf{,.bak.$(date +%F)}
```

The package manager won't help you if you broke a config file at 3 a.m. - your backup will.

**7. Use language-level isolation for language packages.** Python: `python3 -m venv` or `uv`. Node.js: `nvm` for versions, `npm`/`pnpm` for project deps. Don't let `pip install --user` pollute the system Python that other distro packages depend on.

**8. Read what you are about to do.** `apt`/`dnf` print the full plan before they execute. If the list of packages it wants to remove is larger than expected, *stop* and figure out why before saying yes. The vast majority of "I broke my system with apt" stories begin with someone hitting `y` on a prompt that mentioned removing 200 packages.

# Summary

The map of package management you should leave with:

- **A package** is files plus metadata; the manager exists to keep that mapping consistent in a database that survives reboots.
- **High-level tools** (`apt`, `dnf`, `pacman`) run the dependency solver and talk to repositories. **Low-level tools** (`dpkg`, `rpm`) operate on one file at a time.
- The hard part of a package manager is **dependency resolution**, and you avoid most pain by always going through the high-level tool.
- **Repositories** are signed file trees, and the trust chain goes from the GPG keys you ship with the system through `Release` / `repomd.xml` down to each individual `.deb` / `.rpm`.
- **Source builds** are the fallback for when the repo version is wrong; pay the upgrade tax knowingly.
- **Snap / Flatpak / AppImage** solve the "ship to many distros" problem at the cost of disk and integration; great for desktop apps, niche for servers.

Further reading:

- Debian package management reference - <https://www.debian.org/doc/manuals/debian-reference/ch02>
- DNF documentation - <https://dnf.readthedocs.io/>
- Arch Pacman/Rosetta - <https://wiki.archlinux.org/title/Pacman/Rosetta> (cross-distro command translator)
- RPM packaging guide - <https://rpm-packaging-guide.github.io/>

Next in the series:

- *Linux process and resource management* - cgroups, `ps`, `top`, `nice`, OOM killer.
- *Linux user management* - users, groups, sudoers, PAM.

Once you can configure a mirror, lock a version, debug a dependency conflict, and decide between distro packages, source builds and Flatpak on the merits, you have stopped fighting the package manager and started using it.
