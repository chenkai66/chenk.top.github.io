"""
Figures for Cloud Computing Chapter 05: Security and Privacy Protection.

Style: clean whitegrid, color palette
  Blue   #2563eb
  Purple #7c3aed
  Green  #10b981
  Amber  #f59e0b

Generates 5 figures and saves them to BOTH the EN and ZH asset folders so the
two language versions stay in sync.

Figures:
    fig1_shared_responsibility   Provider vs customer responsibility split
                                 across IaaS / PaaS / SaaS, with stacked
                                 layers from physical up to data.
    fig2_iam_model               Users, groups, roles, policies and how
                                 permissions flow from policies through
                                 roles down to identities.
    fig3_encryption_layers       The three states of data and how each is
                                 protected (at rest, in transit, in use)
                                 with example mechanisms.
    fig4_zero_trust              Zero trust architecture: every request
                                 routed through a policy decision point
                                 with identity, device, context signals.
    fig5_compliance_frameworks   SOC 2, HIPAA, GDPR, PCI DSS, ISO 27001
                                 compared on scope, geography and the
                                 control families they emphasise.

Usage:
    python3 scripts/figures/cloud-computing/05-security.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "cloud-computing" / "security-privacy"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "cloud-computing" / "security-privacy"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = "#ffffff",
    ec: str = C_DARK,
    text_color: str | None = None,
    fontsize: float = 10.0,
    weight: str = "normal",
    radius: float = 0.04,
    lw: float = 1.4,
    zorder: int = 3,
) -> None:
    """Draw a rounded rectangle with centred text."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize,
        color=text_color or C_DARK,
        weight=weight,
        zorder=zorder + 1,
    )


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = C_GRAY,
    lw: float = 1.6,
    style: str = "-|>",
    mutation_scale: float = 14,
    zorder: int = 2,
    connectionstyle: str = "arc3,rad=0",
) -> None:
    a = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        color=color,
        linewidth=lw,
        zorder=zorder,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1: Shared Responsibility Model
# ---------------------------------------------------------------------------
def fig1_shared_responsibility() -> None:
    """Stacked layers showing who secures what across IaaS / PaaS / SaaS."""
    fig, ax = plt.subplots(figsize=(11, 6.8))

    # 6 layers from bottom (physical) to top (data)
    layers = [
        "Physical Security",
        "Network Infrastructure",
        "Hypervisor",
        "OS / Runtime",
        "Application",
        "Data & Access",
    ]
    # responsibility per (layer, model) -> "P" provider / "C" customer / "S" shared
    # Models: IaaS, PaaS, SaaS
    matrix = [
        ["P", "P", "P"],  # Physical
        ["P", "P", "P"],  # Network
        ["P", "P", "P"],  # Hypervisor
        ["C", "P", "P"],  # OS/Runtime
        ["C", "C", "P"],  # Application
        ["C", "C", "C"],  # Data
    ]
    models = ["IaaS\n(EC2, GCE)", "PaaS\n(App Engine)", "SaaS\n(Gmail, Salesforce)"]

    layer_h = 0.85
    layer_gap = 0.15
    col_w = 2.2
    col_gap = 0.7
    x0 = 1.4

    for i, layer in enumerate(layers):
        y = i * (layer_h + layer_gap)
        # Layer label on the left
        ax.text(
            x0 - 0.25, y + layer_h / 2, layer,
            ha="right", va="center",
            fontsize=11, color=C_DARK, weight="semibold",
        )
        for j in range(3):
            x = x0 + j * (col_w + col_gap)
            kind = matrix[i][j]
            if kind == "P":
                fc = "#dbeafe"
                ec = C_BLUE
                label = "Provider"
                tc = C_BLUE
            else:
                fc = "#fef3c7"
                ec = C_AMBER
                label = "Customer"
                tc = "#92400e"
            _box(ax, x, y, col_w, layer_h, label, fc=fc, ec=ec, text_color=tc, fontsize=11, weight="semibold", lw=1.6)

    # Column headers
    for j, m in enumerate(models):
        x = x0 + j * (col_w + col_gap) + col_w / 2
        ax.text(
            x, len(layers) * (layer_h + layer_gap) + 0.05, m,
            ha="center", va="bottom",
            fontsize=12, color=C_DARK, weight="bold",
        )

    # Vertical arrow legend "more customer responsibility upward"
    arrow_x = x0 + 3 * (col_w + col_gap) + 0.3
    _arrow(
        ax,
        (arrow_x, 0.1),
        (arrow_x, len(layers) * (layer_h + layer_gap) - 0.05),
        color=C_DARK, lw=1.8, mutation_scale=18,
    )
    ax.text(
        arrow_x + 0.25, len(layers) * (layer_h + layer_gap) / 2,
        "More customer\nresponsibility\nas you move up",
        ha="left", va="center",
        fontsize=10, color=C_DARK, style="italic",
    )

    # Legend
    handles = [
        mpatches.Patch(facecolor="#dbeafe", edgecolor=C_BLUE, label="Provider responsibility"),
        mpatches.Patch(facecolor="#fef3c7", edgecolor=C_AMBER, label="Customer responsibility"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=True, facecolor="white", edgecolor=C_LIGHT)

    ax.set_xlim(0, arrow_x + 2.2)
    ax.set_ylim(-0.3, len(layers) * (layer_h + layer_gap) + 0.6)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Shared Responsibility Model: Who Secures What", fontsize=14, weight="bold", color=C_DARK, pad=14)
    ax.grid(False)

    _save(fig, "fig1_shared_responsibility")


# ---------------------------------------------------------------------------
# Figure 2: IAM model -- users, groups, roles, policies
# ---------------------------------------------------------------------------
def fig2_iam_model() -> None:
    """How identities, groups, roles and policies interlock."""
    fig, ax = plt.subplots(figsize=(11, 6.6))

    # Column 1: Identities (users + service accounts)
    id_items = [
        ("Alice (admin)", C_BLUE),
        ("Bob (developer)", C_BLUE),
        ("Carol (analyst)", C_BLUE),
        ("ci-bot (service)", C_PURPLE),
    ]
    # Column 2: Groups
    groups = [
        ("Admins", C_BLUE, [0]),
        ("Developers", C_BLUE, [1]),
        ("ReadOnly", C_BLUE, [2]),
    ]
    # Column 3: Roles
    roles = [
        ("AdminRole", C_AMBER, [0], [0]),         # group_idx, policy_idx
        ("DeveloperRole", C_AMBER, [1], [1, 2]),
        ("ReadOnlyRole", C_AMBER, [2], [2]),
        ("CIBuildRole", C_AMBER, [], [1]),         # ci-bot via trust, not group
    ]
    # Column 4: Policies
    policies = [
        ("AdminAccess\n(*:*)", C_RED),
        ("DeployS3\n(s3:Put, s3:Get)", C_GREEN),
        ("ReadOnly\n(*:Get, *:List)", C_GREEN),
    ]

    col_x = [0.4, 3.2, 6.0, 8.8]
    col_w = 2.2
    box_h = 0.55
    gap = 0.25

    # Headers
    headers = ["Identities", "Groups", "Roles", "Policies"]
    for i, h in enumerate(headers):
        ax.text(
            col_x[i] + col_w / 2, 5.3, h,
            ha="center", va="bottom",
            fontsize=12, color=C_DARK, weight="bold",
        )

    # Identities
    id_y = []
    for i, (name, c) in enumerate(id_items):
        y = 4.3 - i * (box_h + gap)
        id_y.append(y + box_h / 2)
        fc = "#dbeafe" if c == C_BLUE else "#ede9fe"
        _box(ax, col_x[0], y, col_w, box_h, name, fc=fc, ec=c, fontsize=10, weight="semibold")

    # Groups
    grp_y = []
    for i, (name, c, _) in enumerate(groups):
        y = 4.3 - i * (box_h + gap)
        grp_y.append(y + box_h / 2)
        _box(ax, col_x[1], y, col_w, box_h, name, fc="#dbeafe", ec=c, fontsize=10, weight="semibold")

    # Roles
    role_y = []
    for i, (name, c, _, _) in enumerate(roles):
        y = 4.6 - i * (box_h + gap)
        role_y.append(y + box_h / 2)
        _box(ax, col_x[2], y, col_w, box_h, name, fc="#fef3c7", ec=c, fontsize=10, weight="semibold")

    # Policies
    pol_y = []
    for i, (name, c) in enumerate(policies):
        y = 4.0 - i * (box_h + 0.55)
        pol_y.append(y + box_h / 2)
        fc = "#fee2e2" if c == C_RED else "#d1fae5"
        _box(ax, col_x[3], y, col_w, 0.85, name, fc=fc, ec=c, fontsize=9.5, weight="semibold")

    # Arrows: identities -> groups (membership)
    for i, (_, _, mem_ids) in enumerate(groups):
        for mid in mem_ids:
            _arrow(
                ax,
                (col_x[0] + col_w, id_y[mid]),
                (col_x[1], grp_y[i]),
                color=C_GRAY, lw=1.2,
            )

    # Arrows: ci-bot directly assumes role (trust policy)
    _arrow(
        ax,
        (col_x[0] + col_w, id_y[3]),
        (col_x[2], role_y[3]),
        color=C_PURPLE, lw=1.4, connectionstyle="arc3,rad=-0.15",
    )
    ax.text(
        (col_x[0] + col_w + col_x[2]) / 2, (id_y[3] + role_y[3]) / 2 - 0.3,
        "AssumeRole (trust policy)",
        ha="center", va="top",
        fontsize=8.5, color=C_PURPLE, style="italic",
    )

    # Arrows: groups -> roles
    for i, (_, _, grp_idx, _) in enumerate(roles):
        for gi in grp_idx:
            _arrow(
                ax,
                (col_x[1] + col_w, grp_y[gi]),
                (col_x[2], role_y[i]),
                color=C_GRAY, lw=1.2,
            )

    # Arrows: roles -> policies
    pol_count = [0, 0, 0]
    for i, (_, _, _, pol_idx) in enumerate(roles):
        for pi in pol_idx:
            _arrow(
                ax,
                (col_x[2] + col_w, role_y[i]),
                (col_x[3], pol_y[pi]),
                color=C_AMBER, lw=1.2,
            )
            pol_count[pi] += 1

    # Bottom annotation
    ax.text(
        5.0, -0.15,
        "Identity --(membership)--> Group --(attached)--> Role --(grants)--> Policy --(allows)--> Action on Resource",
        ha="center", va="top",
        fontsize=10, color=C_DARK, style="italic",
        bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT, boxstyle="round,pad=0.4"),
    )

    ax.set_xlim(0, 11.5)
    ax.set_ylim(-0.7, 6.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("IAM Building Blocks: Identities, Groups, Roles, Policies", fontsize=14, weight="bold", color=C_DARK, pad=14)
    ax.grid(False)

    _save(fig, "fig2_iam_model")


# ---------------------------------------------------------------------------
# Figure 3: Encryption layers -- at rest, in transit, in use
# ---------------------------------------------------------------------------
def fig3_encryption_layers() -> None:
    """Three states of data and how each is encrypted."""
    fig, ax = plt.subplots(figsize=(11.5, 6.4))

    # Three big columns
    states = [
        {
            "title": "At Rest",
            "subtitle": "Data sitting on disk",
            "color": C_BLUE,
            "fc": "#dbeafe",
            "icon": "DISK",
            "mechanisms": [
                "AES-256 (XTS / GCM)",
                "Cloud KMS / HSM",
                "Envelope encryption",
                "Server-side encryption (S3, EBS)",
                "TDE (database)",
            ],
            "threat": "Stolen disks,\nbackups, snapshots",
        },
        {
            "title": "In Transit",
            "subtitle": "Data moving between services",
            "color": C_PURPLE,
            "fc": "#ede9fe",
            "icon": "NET",
            "mechanisms": [
                "TLS 1.2+ / TLS 1.3",
                "mTLS (service mesh)",
                "VPN / IPsec tunnels",
                "Signed requests (SigV4)",
                "Certificate pinning",
            ],
            "threat": "MITM,\npacket sniffing,\nDNS hijack",
        },
        {
            "title": "In Use",
            "subtitle": "Data actively being processed",
            "color": C_GREEN,
            "fc": "#d1fae5",
            "icon": "CPU",
            "mechanisms": [
                "Confidential compute (SGX, SEV)",
                "Trusted execution (Nitro Enclaves)",
                "Homomorphic encryption (limited)",
                "Secure multi-party computation",
                "Memory encryption",
            ],
            "threat": "Memory dumps,\nside-channel,\nrogue admin",
        },
    ]

    col_w = 3.4
    col_gap = 0.4
    x0 = 0.4

    for j, st in enumerate(states):
        x = x0 + j * (col_w + col_gap)

        # Header with icon
        _box(ax, x, 5.0, col_w, 1.0, "", fc=st["fc"], ec=st["color"], lw=2.0, radius=0.06)
        ax.text(x + col_w / 2, 5.7, st["title"],
                ha="center", va="center", fontsize=15, weight="bold", color=st["color"])
        ax.text(x + col_w / 2, 5.25, st["subtitle"],
                ha="center", va="center", fontsize=10, color=C_DARK, style="italic")

        # Mechanisms list
        _box(ax, x, 1.6, col_w, 3.2, "", fc="white", ec=C_LIGHT, lw=1.2, radius=0.05)
        ax.text(x + col_w / 2, 4.55, "Mechanisms",
                ha="center", va="center", fontsize=10.5, weight="semibold", color=C_DARK)
        for k, m in enumerate(st["mechanisms"]):
            ax.text(x + 0.18, 4.15 - k * 0.5, "- " + m,
                    ha="left", va="center", fontsize=10, color=C_DARK)

        # Threat box
        _box(ax, x, 0.1, col_w, 1.2, "", fc="#fee2e2", ec=C_RED, lw=1.4, radius=0.05)
        ax.text(x + col_w / 2, 1.05, "Threats if unprotected",
                ha="center", va="center", fontsize=9.5, weight="semibold", color=C_RED)
        ax.text(x + col_w / 2, 0.55, st["threat"],
                ha="center", va="center", fontsize=9.5, color="#991b1b")

    # Connecting horizontal arrow at top showing data flow
    _arrow(ax, (x0 + col_w, 5.5), (x0 + col_w + col_gap, 5.5), color=C_GRAY, lw=1.8, mutation_scale=18)
    _arrow(ax, (x0 + 2 * col_w + col_gap, 5.5), (x0 + 2 * col_w + 2 * col_gap, 5.5), color=C_GRAY, lw=1.8, mutation_scale=18)

    ax.set_xlim(0, 3 * col_w + 2 * col_gap + 0.8)
    ax.set_ylim(-0.2, 6.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Three States of Data, Three Encryption Strategies", fontsize=14, weight="bold", color=C_DARK, pad=14)
    ax.grid(False)

    _save(fig, "fig3_encryption_layers")


# ---------------------------------------------------------------------------
# Figure 4: Zero Trust Architecture
# ---------------------------------------------------------------------------
def fig4_zero_trust() -> None:
    """Every request goes through PDP; identity + device + context signals."""
    fig, ax = plt.subplots(figsize=(11.5, 6.6))

    # Left: requesters
    requesters = [
        ("User\n(laptop)", C_BLUE, 5.0),
        ("User\n(phone)", C_BLUE, 3.5),
        ("Service\n(microservice)", C_PURPLE, 2.0),
        ("Partner\n(API)", C_AMBER, 0.5),
    ]
    for name, c, y in requesters:
        fc = {COLORS["primary"]: "#dbeafe", COLORS["accent"]: "#ede9fe", COLORS["warning"]: "#fef3c7"}.get(c, "#dbeafe")
        _box(ax, 0.2, y, 1.6, 0.9, name, fc=fc, ec=c, fontsize=10, weight="semibold")

    # Centre: Policy Decision Point
    pdp_x, pdp_y, pdp_w, pdp_h = 4.4, 1.8, 2.6, 3.4
    _box(ax, pdp_x, pdp_y, pdp_w, pdp_h, "", fc="#fef3c7", ec=C_AMBER, lw=2.4, radius=0.08)
    ax.text(pdp_x + pdp_w / 2, pdp_y + pdp_h - 0.35,
            "Policy Decision Point",
            ha="center", va="center", fontsize=12, weight="bold", color="#92400e")
    ax.text(pdp_x + pdp_w / 2, pdp_y + pdp_h - 0.75,
            "(PDP)",
            ha="center", va="center", fontsize=10, color="#92400e", style="italic")

    # PDP rules
    rules = [
        "* authenticate identity (MFA)",
        "* verify device posture",
        "* evaluate context",
        "* check least-privilege policy",
        "* log every decision",
    ]
    for k, r in enumerate(rules):
        ax.text(pdp_x + 0.18, pdp_y + pdp_h - 1.25 - k * 0.42, r,
                ha="left", va="center", fontsize=9.5, color=C_DARK)

    # Top: signal sources feeding PDP
    signals = [
        ("Identity\nProvider", C_BLUE, 4.6),
        ("Device\nPosture", C_GREEN, 5.7),
        ("Threat\nIntel", C_RED, 6.8),
    ]
    for name, c, x in signals:
        fc = {COLORS["primary"]: "#dbeafe", COLORS["success"]: "#d1fae5", COLORS["danger"]: "#fee2e2"}.get(c, "white")
        _box(ax, x, 5.7, 1.0, 0.7, name, fc=fc, ec=c, fontsize=9, weight="semibold")
        _arrow(ax, (x + 0.5, 5.7), (x + 0.5, pdp_y + pdp_h), color=c, lw=1.4)

    # Right: protected resources
    resources = [
        ("Database", C_GREEN, 5.0),
        ("API", C_GREEN, 3.5),
        ("Object Store", C_GREEN, 2.0),
        ("Internal App", C_GREEN, 0.5),
    ]
    for name, c, y in resources:
        _box(ax, 9.6, y, 1.7, 0.9, name, fc="#d1fae5", ec=c, fontsize=10, weight="semibold")

    # Arrows: requester -> PDP (route every request through PDP)
    for _, c, y in requesters:
        # Path: out from box, then into the left side of the PDP
        _arrow(
            ax,
            (1.8, y + 0.45), (pdp_x, pdp_y + pdp_h / 2),
            color=c, lw=1.4,
            connectionstyle="arc3,rad=0.0",
        )

    # Arrows: PDP -> resources (allow / deny)
    for name, c, y in resources:
        _arrow(
            ax,
            (pdp_x + pdp_w, pdp_y + pdp_h / 2), (9.6, y + 0.45),
            color=C_GREEN, lw=1.4,
            connectionstyle="arc3,rad=0.0",
        )

    # Bottom annotation
    ax.text(
        5.85, -0.05,
        "Never trust, always verify -- network location grants nothing; every request is authenticated, authorised and logged.",
        ha="center", va="top",
        fontsize=10, color=C_DARK, style="italic",
        bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT, boxstyle="round,pad=0.4"),
    )

    ax.set_xlim(0, 11.7)
    ax.set_ylim(-0.7, 6.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Zero Trust Architecture: Every Request Goes Through Policy", fontsize=14, weight="bold", color=C_DARK, pad=14)
    ax.grid(False)

    _save(fig, "fig4_zero_trust")


# ---------------------------------------------------------------------------
# Figure 5: Compliance frameworks comparison
# ---------------------------------------------------------------------------
def fig5_compliance_frameworks() -> None:
    """Heatmap-style comparison of frameworks across control families."""
    fig, ax = plt.subplots(figsize=(11.5, 6.4))

    frameworks = ["SOC 2", "HIPAA", "GDPR", "PCI DSS", "ISO 27001"]
    families = [
        "Access Control",
        "Encryption",
        "Audit Logging",
        "Incident Response",
        "Data Subject Rights",
        "Vendor Management",
        "Physical Security",
    ]
    # Coverage strength 0..3 (none, partial, strong, mandatory)
    coverage = np.array([
        # Access  Encryp  Audit   IR      DSR     Vendor  Phys
        [3, 2, 3, 3, 1, 2, 1],   # SOC 2
        [3, 3, 3, 3, 2, 3, 2],   # HIPAA
        [3, 3, 2, 3, 3, 2, 1],   # GDPR
        [3, 3, 3, 3, 1, 2, 3],   # PCI DSS
        [3, 2, 3, 3, 1, 3, 2],   # ISO 27001
    ])

    # Build a custom colormap-like mapping using our palette
    palette = {
        0: "#f1f5f9",  # none
        1: "#bfdbfe",  # partial
        2: "#60a5fa",  # strong
        3: C_BLUE,     # mandatory
    }
    labels_map = {0: "n/a", 1: "partial", 2: "strong", 3: "mandatory"}

    cell_w, cell_h = 1.25, 0.7
    x0, y0 = 1.6, 1.6

    # Cells
    for i in range(len(frameworks)):
        for j in range(len(families)):
            v = coverage[i, j]
            x = x0 + j * cell_w
            y = y0 + (len(frameworks) - 1 - i) * cell_h
            ax.add_patch(Rectangle((x, y), cell_w * 0.95, cell_h * 0.9,
                                   facecolor=palette[v], edgecolor="white", linewidth=1.2))
            ax.text(x + cell_w * 0.475, y + cell_h * 0.45,
                    labels_map[v],
                    ha="center", va="center",
                    fontsize=8.5,
                    color="white" if v >= 2 else C_DARK,
                    weight="semibold" if v >= 2 else "normal")

    # Framework labels (rows)
    for i, fw in enumerate(frameworks):
        y = y0 + (len(frameworks) - 1 - i) * cell_h + cell_h * 0.45
        ax.text(x0 - 0.15, y, fw,
                ha="right", va="center",
                fontsize=11, weight="bold", color=C_DARK)

    # Control family labels (columns) -- rotated
    for j, fam in enumerate(families):
        x = x0 + j * cell_w + cell_w * 0.475
        ax.text(x, y0 + len(frameworks) * cell_h + 0.15, fam,
                ha="center", va="bottom",
                fontsize=10, color=C_DARK, rotation=20, weight="semibold")

    # Side panel: scope summary
    panel_x = x0 + len(families) * cell_w + 0.5
    panel_w = 2.6
    _box(ax, panel_x, y0, panel_w, len(frameworks) * cell_h - 0.05, "", fc=C_BG, ec=C_LIGHT, lw=1.2, radius=0.04)
    ax.text(panel_x + panel_w / 2, y0 + len(frameworks) * cell_h - 0.25,
            "Scope at a glance",
            ha="center", va="top",
            fontsize=10.5, weight="bold", color=C_DARK)
    scopes = [
        ("SOC 2", "Service orgs (US)", C_BLUE),
        ("HIPAA", "US healthcare PHI", C_PURPLE),
        ("GDPR", "EU personal data", C_GREEN),
        ("PCI DSS", "Card payment data", C_AMBER),
        ("ISO 27001", "International ISMS", C_RED),
    ]
    for k, (name, desc, c) in enumerate(scopes):
        ay = y0 + len(frameworks) * cell_h - 0.65 - k * 0.5
        ax.text(panel_x + 0.15, ay, name,
                ha="left", va="center",
                fontsize=9.5, weight="bold", color=c)
        ax.text(panel_x + 1.05, ay, desc,
                ha="left", va="center",
                fontsize=9, color=C_DARK)

    # Legend at bottom
    legend_y = 0.6
    legend_items = [(0, "n/a"), (1, "partial"), (2, "strong"), (3, "mandatory")]
    for k, (v, lbl) in enumerate(legend_items):
        lx = x0 + k * 1.5
        ax.add_patch(Rectangle((lx, legend_y), 0.4, 0.3, facecolor=palette[v], edgecolor="white"))
        ax.text(lx + 0.5, legend_y + 0.15, lbl,
                ha="left", va="center", fontsize=9.5, color=C_DARK)

    ax.set_xlim(0, panel_x + panel_w + 0.4)
    ax.set_ylim(0.2, y0 + len(frameworks) * cell_h + 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Compliance Frameworks: Coverage Across Control Families", fontsize=14, weight="bold", color=C_DARK, pad=14)
    ax.grid(False)

    _save(fig, "fig5_compliance_frameworks")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[Cloud Computing 05: Security] generating figures...")
    fig1_shared_responsibility()
    fig2_iam_model()
    fig3_encryption_layers()
    fig4_zero_trust()
    fig5_compliance_frameworks()
    print("done.")
