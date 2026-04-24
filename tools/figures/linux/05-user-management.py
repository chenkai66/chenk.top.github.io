"""
Figure generation script for Linux Article 05: User Management.

Generates 5 conceptual figures used in both EN and ZH versions of the
article. Each figure is rendered to BOTH article asset folders so the
markdown image references stay in sync across languages.

Figures:
    fig1_passwd_anatomy           Anatomy of one /etc/passwd line, with each
                                  colon-separated field highlighted and
                                  annotated (username, x, UID, GID, GECOS,
                                  home, shell). Includes the parallel
                                  /etc/shadow line and shows the link
                                  between the two files.
    fig2_user_group_relationship  Set/Venn-style diagram of how users belong
                                  to one primary group plus zero-or-more
                                  supplementary groups, including how that
                                  determines access to group-owned
                                  resources.
    fig3_sudo_policy_hierarchy    Layered diagram of how a sudo decision is
                                  made: user request -> /etc/sudoers and
                                  /etc/sudoers.d/* rules -> runas user/group
                                  -> command allowlist -> NOPASSWD/timeout
                                  -> audit log.
    fig4_user_lifecycle_commands  Lifecycle of a Linux account on one row:
                                  useradd -> usermod -> passwd / chage ->
                                  usermod -L / passwd -l -> userdel, with
                                  the files each command touches listed
                                  underneath (passwd / shadow / group /
                                  gshadow / home / mail spool).
    fig5_pam_auth_flow            Stack diagram of how PAM evaluates an
                                  authentication request: auth -> account
                                  -> password -> session, with the typical
                                  modules at each stage and required vs.
                                  sufficient control flags.

Usage:
    python3 scripts/figures/linux/05-user-management.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401  (registers the seaborn style we use)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG_SOFT = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "linux" / "user-management"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linux" / "用户管理"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _no_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


def _box(ax, x, y, w, h, *, fc, ec=None, alpha=1.0, lw=1.2, radius=0.06):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc, edgecolor=ec or fc, linewidth=lw, alpha=alpha,
    ))


def _arrow(ax, x1, y1, x2, y2, *, color=C_GRAY, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, color=color, linewidth=lw,
        mutation_scale=15, shrinkA=2, shrinkB=2,
    ))


# ---------------------------------------------------------------------------
# Figure 1 -- Anatomy of /etc/passwd (and its link to /etc/shadow)
# ---------------------------------------------------------------------------
def fig1_passwd_anatomy() -> None:
    fig, ax = plt.subplots(figsize=(15.5, 7.8))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.8)
    _no_axis(ax)

    ax.text(7.75, 7.4, "Anatomy of an /etc/passwd entry",
            ha="center", va="center",
            fontsize=15.5, fontweight="bold", color=C_DARK)
    ax.text(7.75, 7.0, "Seven colon-separated fields, plus the link to /etc/shadow",
            ha="center", va="center",
            fontsize=10.5, color=C_GRAY, style="italic")

    # The passwd line, broken into 7 cells.
    fields = [
        ("alice",          C_BLUE,   "username",
         "login name;\nmust be unique"),
        ("x",              C_GRAY,   "password\nplaceholder",
         "real hash lives\nin /etc/shadow"),
        ("1001",           C_PURPLE, "UID",
         "0 = root\n1-999 = system\n1000+ = human"),
        ("1001",           C_PURPLE, "GID",
         "primary group;\nlook up\nin /etc/group"),
        ("Alice Wang",     C_AMBER,  "GECOS",
         "comment /\nfull name;\noptional"),
        ("/home/alice",    C_GREEN,  "home dir",
         "$HOME at login;\ntemplate from\n/etc/skel"),
        ("/bin/bash",      C_GREEN,  "login shell",
         "/sbin/nologin\nto disable\ninteractive login"),
    ]

    cell_w = 2.05
    start_x = 7.75 - (len(fields) * cell_w) / 2
    y_band = 4.75
    h_band = 0.95

    for i, (text, color, label, note) in enumerate(fields):
        x = start_x + i * cell_w
        _box(ax, x, y_band, cell_w * 0.94, h_band, fc=color, ec="white", lw=1.5)
        ax.text(x + cell_w * 0.47, y_band + h_band / 2, text,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white", family="monospace")
        # Header label above
        ax.text(x + cell_w * 0.47, y_band + h_band + 0.32, label,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=color)
        # Note below
        ax.text(x + cell_w * 0.47, y_band - 0.55, note,
                ha="center", va="top", fontsize=8.6, color=C_DARK,
                family="monospace")
        # Colon separators
        if i < len(fields) - 1:
            ax.text(x + cell_w * 0.94 + 0.02, y_band + h_band / 2, ":",
                    ha="left", va="center", fontsize=18,
                    fontweight="bold", color=C_DARK, family="monospace")

    # Link to /etc/shadow
    shadow_y = 1.55
    _box(ax, 2.0, shadow_y - 0.05, 11.5, 0.95,
         fc=C_BG_SOFT, ec=C_GRAY, lw=1.0)
    ax.text(7.75, shadow_y + 0.65,
            "/etc/shadow  (root-only)  -- linked by username",
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=C_DARK)
    ax.text(7.75, shadow_y + 0.18,
            "alice : $6$rsalt$Wq...hash : 19840 : 0 : 90 : 7 : 30 : :",
            ha="center", va="center", fontsize=11,
            color=C_RED, family="monospace")

    # Arrow from passwd "x" cell down to shadow row
    x_cell_center = start_x + 1 * cell_w + cell_w * 0.47
    _arrow(ax, x_cell_center, y_band - 1.45, x_cell_center, shadow_y + 0.95,
           color=C_GRAY, lw=1.4)
    ax.text(x_cell_center + 0.25, (y_band - 1.45 + shadow_y + 0.95) / 2,
            "the 'x' here means\n'see /etc/shadow'",
            ha="left", va="center", fontsize=8.8, color=C_GRAY, style="italic")

    # Footer
    ax.text(7.75, 0.55,
            "Tip: never edit these files by hand -- use `vipw` / `vigr`,"
            " or the wrappers `useradd` / `usermod` / `passwd`.",
            ha="center", va="center", fontsize=9.5,
            color=C_DARK, style="italic")

    fig.tight_layout()
    _save(fig, "fig1_passwd_anatomy")


# ---------------------------------------------------------------------------
# Figure 2 -- Users, primary group, and supplementary groups
# ---------------------------------------------------------------------------
def fig2_user_group_relationship() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.6)
    _no_axis(ax)

    ax.text(6.5, 7.2, "One user, one primary group, many supplementary groups",
            ha="center", va="center",
            fontsize=15, fontweight="bold", color=C_DARK)
    ax.text(6.5, 6.78,
            "The primary group decides default ownership;"
            " supplementary groups grant extra access",
            ha="center", va="center",
            fontsize=10.5, color=C_GRAY, style="italic")

    # Three users on the left
    users = [
        ("alice",  5.2, C_BLUE),
        ("bob",    3.6, C_PURPLE),
        ("carol",  2.0, C_GREEN),
    ]
    for name, y, color in users:
        ax.add_patch(Circle((1.1, y), 0.42, facecolor=color, edgecolor="white", lw=2))
        ax.text(1.1, y, name[0].upper(), ha="center", va="center",
                fontsize=15, fontweight="bold", color="white")
        # Label to the right of the circle, not below (avoids being mistaken
        # for the next user's label).
        ax.text(1.7, y, name, ha="left", va="center",
                fontsize=10.5, fontweight="bold", color=C_DARK,
                family="monospace")

    # Primary group column
    ax.text(4.5, 6.15, "Primary group (GID in /etc/passwd)",
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=C_DARK)
    primary = [
        ("alice",      5.2, C_BLUE),
        ("developers", 3.6, C_PURPLE),
        ("carol",      2.0, C_GREEN),
    ]
    for gname, y, color in primary:
        _box(ax, 3.65, y - 0.32, 1.7, 0.65, fc=color, ec="white", lw=1.5, alpha=0.85)
        ax.text(4.5, y, gname, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="white",
                family="monospace")

    # Arrows users -> primary group
    for (_, uy, _), (_, gy, color) in zip(users, primary):
        _arrow(ax, 2.55, uy, 3.65, gy, color=color, lw=1.6)

    # Supplementary groups column (a shared pool)
    ax.text(9.5, 6.15, "Supplementary groups (members in /etc/group)",
            ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=C_DARK)

    sup_groups = [
        ("docker",     7.2, 5.2, C_AMBER, ["alice", "bob"]),
        ("developers", 7.2, 3.6, C_PURPLE, ["alice", "bob"]),
        ("sudo",       7.2, 2.0, C_RED,   ["alice"]),
    ]
    for gname, x, y, color, members in sup_groups:
        _box(ax, x, y - 0.42, 4.4, 0.85, fc=C_BG_SOFT, ec=color, lw=1.6)
        ax.text(x + 0.12, y + 0.12, gname, ha="left", va="center",
                fontsize=11, fontweight="bold", color=color,
                family="monospace")
        ax.text(x + 0.12, y - 0.22, "members: " + ", ".join(members),
                ha="left", va="center", fontsize=9, color=C_DARK,
                family="monospace")

    # Arrows users -> supplementary groups (only for those who belong)
    membership_map = {
        "alice": ["docker", "developers", "sudo"],
        "bob":   ["docker", "developers"],
        "carol": [],
    }
    sup_y = {g[0]: g[2] for g in sup_groups}

    for name, uy, color in users:
        for g in membership_map[name]:
            _arrow(ax, 5.35, uy, 7.2, sup_y[g], color=color, lw=1.0, style="->")

    # Bottom legend / takeaway
    _box(ax, 1.0, 0.35, 11.0, 0.95, fc=C_BG_SOFT, ec=C_GRAY, lw=1.0, radius=0.04)
    ax.text(6.5, 0.95,
            "alice has primary group `alice` and supplementary"
            " groups {docker, developers, sudo}",
            ha="center", va="center", fontsize=10.2,
            color=C_DARK, family="monospace")
    ax.text(6.5, 0.6,
            "files alice creates are owned by alice:alice"
            " unless the parent dir has the SGID bit set",
            ha="center", va="center", fontsize=9.5,
            color=C_GRAY, style="italic")

    fig.tight_layout()
    _save(fig, "fig2_user_group_relationship")


# ---------------------------------------------------------------------------
# Figure 3 -- The sudo policy hierarchy
# ---------------------------------------------------------------------------
def fig3_sudo_policy_hierarchy() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.2)
    _no_axis(ax)

    ax.text(6.5, 7.85,
            "How sudo decides whether to run your command",
            ha="center", va="center", fontsize=15.5,
            fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.45,
            "Each rule must match all five dimensions before the command runs",
            ha="center", va="center", fontsize=10.5,
            color=C_GRAY, style="italic")

    # User input on the left
    _box(ax, 0.4, 3.4, 2.4, 1.2, fc=C_BLUE, ec="white", lw=1.5)
    ax.text(1.6, 4.2, "alice@host", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
            family="monospace")
    ax.text(1.6, 3.75, "$ sudo systemctl\n  restart nginx",
            ha="center", va="center", fontsize=9.5,
            color="white", family="monospace")

    # Five dimensions, vertically stacked
    layers = [
        ("WHO",      "user/%group entry",
         "alice  or  %ops",
         C_BLUE,    6.45),
        ("WHERE",    "host pattern",
         "ALL  or  web01,web02",
         C_PURPLE,  5.35),
        ("AS WHOM",  "(runas_user:runas_group)",
         "(ALL:ALL)  or  (root)",
         C_AMBER,   4.25),
        ("WHAT",     "command allowlist",
         "/usr/bin/systemctl restart nginx",
         C_GREEN,   3.15),
        ("OPTIONS",  "tags & defaults",
         "NOPASSWD: , timestamp_timeout, log_output",
         C_GRAY,    2.05),
    ]

    for label, sub, example, color, y in layers:
        _box(ax, 4.2, y - 0.42, 7.6, 0.9, fc=C_BG_SOFT, ec=color, lw=1.6)
        ax.text(4.4, y + 0.12, label, ha="left", va="center",
                fontsize=11, fontweight="bold", color=color,
                family="monospace")
        ax.text(5.3, y + 0.12, sub, ha="left", va="center",
                fontsize=9.5, color=C_DARK)
        ax.text(4.4, y - 0.22, example, ha="left", va="center",
                fontsize=9.3, color=C_DARK, family="monospace",
                style="italic")

    # Arrow from user into the stack (top), then between layers
    _arrow(ax, 2.8, 4.0, 4.2, 6.3, color=C_BLUE, lw=1.6)
    for i in range(len(layers) - 1):
        y_top = layers[i][4] - 0.42
        y_bot = layers[i + 1][4] + 0.48
        _arrow(ax, 8.0, y_top, 8.0, y_bot, color=C_GRAY, lw=1.2)

    # Outcome on the right
    _box(ax, 11.4, 3.4, 1.4, 1.2, fc=C_GREEN, ec="white", lw=1.5)
    ax.text(12.1, 4.2, "ALLOW", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
            family="monospace")
    ax.text(12.1, 3.75, "audit log\n+ exec", ha="center", va="center",
            fontsize=9, color="white")

    _arrow(ax, 11.8, layers[-1][4], 11.4, 4.0, color=C_GREEN, lw=1.6)

    # Footer note about file precedence
    _box(ax, 0.6, 0.35, 11.8, 1.15, fc=C_BG_SOFT, ec=C_GRAY, lw=1.0, radius=0.04)
    ax.text(6.5, 1.18,
            "Rule sources, evaluated in order:",
            ha="center", va="center", fontsize=10.2,
            fontweight="bold", color=C_DARK)
    ax.text(6.5, 0.78,
            "/etc/sudoers   ->   /etc/sudoers.d/*   (lexical order)"
            "   ->   group memberships   ->   defaults",
            ha="center", va="center", fontsize=9.8,
            color=C_DARK, family="monospace")
    ax.text(6.5, 0.45,
            "always edit with `visudo` so a syntax error cannot lock you out",
            ha="center", va="center", fontsize=9.2,
            color=C_GRAY, style="italic")

    fig.tight_layout()
    _save(fig, "fig3_sudo_policy_hierarchy")


# ---------------------------------------------------------------------------
# Figure 4 -- The user lifecycle: useradd -> usermod -> passwd -> userdel
# ---------------------------------------------------------------------------
def fig4_user_lifecycle_commands() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.6)
    _no_axis(ax)

    ax.text(7.0, 7.2, "The lifecycle of a Linux account",
            ha="center", va="center", fontsize=15.5,
            fontweight="bold", color=C_DARK)
    ax.text(7.0, 6.78,
            "Each command edits a specific set of files -- knowing which"
            " makes recovery and auditing trivial",
            ha="center", va="center", fontsize=10.5,
            color=C_GRAY, style="italic")

    stages = [
        ("useradd",
         "create",
         ["/etc/passwd", "/etc/shadow",
          "/etc/group", "/home/<user>",
          "copies /etc/skel"],
         C_BLUE),
        ("usermod",
         "modify\n(name, shell, groups,\nhome, lock/unlock)",
         ["/etc/passwd", "/etc/shadow",
          "/etc/group", "/etc/gshadow"],
         C_PURPLE),
        ("passwd / chage",
         "set password\n& aging policy",
         ["/etc/shadow",
          "min/max/warn/inactive",
          "expire date"],
         C_AMBER),
        ("usermod -L\n(or passwd -l)",
         "lock\n(soft delete)",
         ["/etc/shadow",
          "(prepends '!' to hash;\nlogins blocked,\nfiles preserved)"],
         C_GRAY),
        ("userdel [-r]",
         "delete\n(-r removes $HOME)",
         ["/etc/passwd", "/etc/shadow",
          "/etc/group", "/etc/gshadow",
          "/home/<user> (with -r)",
          "/var/spool/mail/<user>"],
         C_RED),
    ]

    n = len(stages)
    cell_w = 2.5
    gap = 0.3
    total = n * cell_w + (n - 1) * gap
    start_x = (14 - total) / 2
    y_top = 5.4
    box_h = 1.1

    for i, (cmd, action, files, color) in enumerate(stages):
        x = start_x + i * (cell_w + gap)
        # Command box
        _box(ax, x, y_top, cell_w, box_h, fc=color, ec="white", lw=1.6)
        ax.text(x + cell_w / 2, y_top + box_h / 2, cmd,
                ha="center", va="center", fontsize=11.5,
                fontweight="bold", color="white", family="monospace")
        # Action label
        ax.text(x + cell_w / 2, y_top - 0.35, action,
                ha="center", va="top", fontsize=9.5,
                color=C_DARK, fontweight="bold")
        # Files touched
        files_text = "\n".join(files)
        _box(ax, x, 1.45, cell_w, 2.5, fc=C_BG_SOFT, ec=color, lw=1.2)
        ax.text(x + cell_w / 2, 3.85, "touches",
                ha="center", va="top", fontsize=9, color=color,
                fontweight="bold", style="italic")
        ax.text(x + cell_w / 2, 3.5, files_text,
                ha="center", va="top", fontsize=8.6,
                color=C_DARK, family="monospace")

        # Arrow to the next stage
        if i < n - 1:
            arrow_y = y_top + box_h / 2
            _arrow(ax, x + cell_w + 0.02, arrow_y,
                   x + cell_w + gap - 0.02, arrow_y,
                   color=C_GRAY, lw=1.4)

    # Bottom callout: best practice
    _box(ax, 0.6, 0.25, 12.8, 0.95, fc=C_BG_SOFT, ec=C_GRAY, lw=1.0, radius=0.04)
    ax.text(7.0, 0.85,
            "Best practice: lock first (`usermod -L`), wait for any open"
            " files / cron jobs to drain, then `userdel -r`.",
            ha="center", va="center", fontsize=10,
            color=C_DARK, fontweight="bold")
    ax.text(7.0, 0.5,
            "deleting straight away leaves orphaned files owned by a"
            " bare UID, which the next new account silently inherits.",
            ha="center", va="center", fontsize=9.2,
            color=C_GRAY, style="italic")

    fig.tight_layout()
    _save(fig, "fig4_user_lifecycle_commands")


# ---------------------------------------------------------------------------
# Figure 5 -- PAM authentication flow
# ---------------------------------------------------------------------------
def fig5_pam_auth_flow() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 8.0)
    _no_axis(ax)

    ax.text(6.75, 7.65,
            "How PAM evaluates `login` (or `sshd`, `sudo`, `su`...)",
            ha="center", va="center", fontsize=15.5,
            fontweight="bold", color=C_DARK)
    ax.text(6.75, 7.25,
            "Four stacks run in order; each stack is a list of modules"
            " gated by control flags",
            ha="center", va="center", fontsize=10.5,
            color=C_GRAY, style="italic")

    # Left: requesting service
    _box(ax, 0.4, 3.3, 2.0, 1.4, fc=C_BLUE, ec="white", lw=1.5)
    ax.text(1.4, 4.3, "sshd", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white",
            family="monospace")
    ax.text(1.4, 3.9, "(or su, sudo,\nlogin, gdm...)",
            ha="center", va="center", fontsize=8.8,
            color="white")
    ax.text(1.4, 3.05,
            "reads\n/etc/pam.d/sshd",
            ha="center", va="top", fontsize=8.5, color=C_DARK,
            style="italic", family="monospace")

    # Four stacks, horizontally laid out
    stacks = [
        ("auth",
         "Are you who you\nclaim to be?",
         ["pam_unix.so",
          "pam_sss.so / pam_ldap.so",
          "pam_google_authenticator.so"],
         C_BLUE),
        ("account",
         "Is this account\nallowed right now?",
         ["pam_unix.so  (expiry)",
          "pam_access.so  (host/time)",
          "pam_nologin.so"],
         C_PURPLE),
        ("password",
         "On change, is the\nnew secret strong?",
         ["pam_pwquality.so  (rules)",
          "pam_pwhistory.so  (history)",
          "pam_unix.so  (write)"],
         C_AMBER),
        ("session",
         "Set up the user's\nworking environment.",
         ["pam_limits.so  (ulimit)",
          "pam_systemd.so  (slice)",
          "pam_mkhomedir.so",
          "pam_lastlog.so"],
         C_GREEN),
    ]

    n = len(stacks)
    cell_w = 2.5
    gap = 0.25
    total = n * cell_w + (n - 1) * gap
    start_x = 3.2
    if start_x + total > 13.0:
        start_x = 13.0 - total
    y_top = 5.7
    head_h = 0.8

    for i, (name, q, modules, color) in enumerate(stacks):
        x = start_x + i * (cell_w + gap)
        # Stack header
        _box(ax, x, y_top, cell_w, head_h, fc=color, ec="white", lw=1.5)
        ax.text(x + cell_w / 2, y_top + head_h / 2, name,
                ha="center", va="center", fontsize=12,
                fontweight="bold", color="white", family="monospace")
        # Question
        ax.text(x + cell_w / 2, y_top - 0.05, q,
                ha="center", va="top", fontsize=9, color=C_DARK,
                style="italic")
        # Modules
        _box(ax, x, 2.3, cell_w, 2.6, fc=C_BG_SOFT, ec=color, lw=1.2)
        ax.text(x + cell_w / 2, 4.75, "typical modules",
                ha="center", va="top", fontsize=8.8,
                color=color, fontweight="bold", style="italic")
        ax.text(x + cell_w / 2, 4.4, "\n".join(modules),
                ha="center", va="top", fontsize=8.6,
                color=C_DARK, family="monospace")

        # Arrow to next stack header
        if i < n - 1:
            ay = y_top + head_h / 2
            _arrow(ax, x + cell_w + 0.02, ay,
                   x + cell_w + gap - 0.02, ay,
                   color=C_GRAY, lw=1.4)

    # Arrow from sshd to first stack
    _arrow(ax, 2.4, 4.0, start_x, y_top + head_h / 2,
           color=C_BLUE, lw=1.6)

    # Bottom: control flags legend
    _box(ax, 0.6, 0.4, 12.3, 1.5, fc=C_BG_SOFT, ec=C_GRAY, lw=1.0, radius=0.04)
    ax.text(6.75, 1.6, "Control flags decide how each module's result is combined",
            ha="center", va="center", fontsize=10.2,
            fontweight="bold", color=C_DARK)
    flag_lines = [
        ("required",   "must succeed; failure is remembered, but the rest of the stack still runs",   C_BLUE),
        ("requisite",  "must succeed; on failure the stack stops immediately",                         C_PURPLE),
        ("sufficient", "if it succeeds AND no earlier `required` failed, the stack passes right away", C_GREEN),
        ("optional",   "result is ignored unless it is the only module in the stack",                  C_GRAY),
    ]
    for i, (flag, desc, color) in enumerate(flag_lines):
        y = 1.25 - i * 0.22
        ax.text(1.0, y, flag, ha="left", va="center",
                fontsize=9, fontweight="bold", color=color,
                family="monospace")
        ax.text(2.6, y, "  " + desc, ha="left", va="center",
                fontsize=9, color=C_DARK)

    fig.tight_layout()
    _save(fig, "fig5_pam_auth_flow")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Linux Article 05 figures (User Management)...")
    fig1_passwd_anatomy()
    fig2_user_group_relationship()
    fig3_sudo_policy_hierarchy()
    fig4_user_lifecycle_commands()
    fig5_pam_auth_flow()
    print("Done.")


if __name__ == "__main__":
    main()
