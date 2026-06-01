# C-d# =============================================================================
# SIMULACIÓN DE PROPAGACIÓN DINÁMICA DE GRIETAS EN 2D
# MDF + RUNGE KUTTA 4 + ZONA DE DAÑO
#
# Material: ASTM A36
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

# =============================================================================
# 1. PROPIEDADES DEL MATERIAL ASTM A36
# =============================================================================

E = 200e9
rho = 7850
nu = 0.26
sigma_crit = 250e6

# Velocidad de onda
c = np.sqrt(E / rho)

print("====================================")
print("PROPIEDADES DEL MATERIAL ASTM A36")
print("====================================")
print(f"Velocidad de onda: {c:.2f} m/s")
print()

# =============================================================================
# 2. GEOMETRÍA Y DISCRETIZACIÓN
# =============================================================================

Lx = 1.0
Ly = 1.0

Nx = 101
Ny = 101

dx = Lx / (Nx - 1)
dy = Ly / (Ny - 1)

dt = 1.0e-6
Nt = 600

print("====================================")
print("PARÁMETROS NUMÉRICOS")
print("====================================")
print(f"dx = {dx:.5f} m")
print(f"dy = {dy:.5f} m")
print(f"dt = {dt:.2e} s")
print(f"Número de pasos: {Nt}")
print()

# =============================================================================
# 3. CREACIÓN DE LA MALLA
# =============================================================================

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

X, Y = np.meshgrid(x, y)

# =============================================================================
# 4. VARIABLES PRINCIPALES
# =============================================================================

u = np.zeros((Ny, Nx))     # desplazamiento
v = np.zeros((Ny, Nx))     # velocidad

# =============================================================================
# 5. CONDICIÓN INICIAL
# =============================================================================

xc = 0.25
yc = 0.50

sigma = 0.03

u = np.exp(-((X - xc)**2 + (Y - yc)**2) / sigma**2)

# =============================================================================
# 6. GRIETA Y DAÑO
# =============================================================================

crack = np.zeros((Ny, Nx), dtype=bool)

# Variable de daño
damage = np.zeros((Ny, Nx))

j_crack = Ny // 2

# Grieta inicial
for i in range(30, 50):

    crack[j_crack, i] = True

    damage[
        j_crack - 1:j_crack + 2,
        i - 1:i + 2
    ] = 1.0

# Punta inicial
crack_tip_i = 50
crack_tip_j = j_crack

# =============================================================================
# 7. LAPLACIANA CON DAÑO
# =============================================================================

def laplacian(u):

    lap = np.zeros_like(u)

    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):

            # Material roto
            if damage[j, i] >= 1.0:
                continue

            d2udx2 = (
                u[j, i + 1]
                - 2 * u[j, i]
                + u[j, i - 1]
            ) / dx**2

            d2udy2 = (
                u[j + 1, i]
                - 2 * u[j, i]
                + u[j - 1, i]
            ) / dy**2

            # Rigidez degradada
            c_eff = c * (1.0 - damage[j, i])

            lap[j, i] = (
                c_eff**2
                * (d2udx2 + d2udy2)
            )

    return lap

# =============================================================================
# 8. CÁLCULO DE ESFUERZO
# =============================================================================

def compute_stress(u, i, j, dx, E):

    dudx = (
        u[j, i + 1]
        - u[j, i - 1]
    ) / (2 * dx)

    sigma = E * dudx

    return abs(sigma)

# =============================================================================
# 9. CONFIGURACIÓN DE FIGURA
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 7))

# Colormap más contrastante
cmap = plt.cm.plasma.copy()

# La grieta será negra
cmap.set_bad(color='black')

im = ax.imshow(
    u,
    extent=[0, Lx, 0, Ly],
    cmap=cmap,
    origin='lower',
    vmin=-1,
    vmax=1,
    animated=True
)

cb = plt.colorbar(im)
cb.set_label(
    'Desplazamiento',
    fontsize=12
)

ax.set_title(
    "Propagación Dinámica de Grieta\n"
    "MDF + RK4 + Zona de Fractura",
    fontsize=14,
    fontweight='bold'
)

ax.set_xlabel("x [m]", fontsize=12)
ax.set_ylabel("y [m]", fontsize=12)

# Fondo oscuro para mayor contraste
ax.set_facecolor('black')

# =============================================================================
# 10. FUNCIÓN UPDATE
# =============================================================================

def update(frame):

    global u, v
    global crack_tip_i
    global damage

    # -------------------------------------------------------------------------
    # RUNGE-KUTTA 4
    # -------------------------------------------------------------------------

    def du_dt(v):
        return v

    def dv_dt(u):
        return laplacian(u)

    # k1
    k1u = du_dt(v)
    k1v = dv_dt(u)

    # k2
    k2u = du_dt(v + 0.5 * dt * k1v)
    k2v = dv_dt(u + 0.5 * dt * k1u)

    # k3
    k3u = du_dt(v + 0.5 * dt * k2v)
    k3v = dv_dt(u + 0.5 * dt * k2u)

    # k4
    k4u = du_dt(v + dt * k3v)
    k4v = dv_dt(u + dt * k3u)

    # Actualización temporal
    u += (dt / 6.0) * (
        k1u
        + 2 * k2u
        + 2 * k3u
        + k4u
    )

    v += (dt / 6.0) * (
        k1v
        + 2 * k2v
        + 2 * k3v
        + k4v
    )

    # -------------------------------------------------------------------------
    # CONDICIONES DE FRONTERA
    # -------------------------------------------------------------------------

    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0

    # -------------------------------------------------------------------------
    # PROPAGACIÓN DE GRIETA
    # -------------------------------------------------------------------------

    if crack_tip_i < Nx - 4:

        sigma_tip = compute_stress(
            u,
            crack_tip_i,
            crack_tip_j,
            dx,
            E
        )

        # Criterio de fractura
        if sigma_tip >= sigma_crit:

            crack_tip_i += 1

            crack[
                crack_tip_j,
                crack_tip_i
            ] = True

            # Zona dañada más gruesa
            damage[
                crack_tip_j - 2:crack_tip_j + 3,
                crack_tip_i - 2:crack_tip_i + 3
            ] = 1.0

            print(
                f"Paso {frame:04d} | "
                f"Grieta propagada en x = {crack_tip_i}"
            )

    # -------------------------------------------------------------------------
    # VISUALIZACIÓN
    # -------------------------------------------------------------------------

    u_plot = np.copy(u)

    # La zona dañada aparece negra
    u_plot = np.ma.array(
        u_plot,
        mask=(damage > 0.9)
    )

    im.set_array(u_plot)

    return [im]

# =============================================================================
# 11. ANIMACIÓN
# =============================================================================

ani = FuncAnimation(
    fig,
    update,
    frames=Nt,
    interval=20,
    blit=True
)

# =============================================================================
# 12. RESULTADOS
# =============================================================================

plt.tight_layout()
plt.show()

# =============================================================================
# FIN DEL PROGRAMA
# =============================================================================igo-del-programa-en-python-
