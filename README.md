calculo-diferencial-circuito-rc/
├── main.py
├── requirements.txt
└── README.md
numpy
scipy
matplotlib
"""
Análisis Diferencial y Simulación Numérica de la Dinámica de Carga y Descarga en Circuitos RC

Autores: 
- Diego Esteban Torres Puentes (dtorres00@unisalle.edu.co)
- Jhon Esteban Bedoya Vargas (jvargas12@unisalle.edu.co)

Programa de Ingeniería de Software - Universidad de La Salle
Asignatura: Cálculo Diferencial
Fecha: Noviembre 2026
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ==========================================
# 1. PARÁMETROS NOMINALES E INCERTIDUMBRES
# ==========================================
R_nominal = 10000.0       # Resistencia en Ohmios (10.0 kΩ)
sigma_R = 100.0           # Incertidumbre de R (±0.1 kΩ)

C_nominal = 100e-6        # Capacitancia en Faradios (100 µF)
sigma_C = 2e-6            # Incertidumbre de C (±2 µF)

V0_nominal = 5.00         # Voltaje de fuente en Voltios (5.00 V)
sigma_V0 = 0.05           # Incertidumbre de V0 (±0.05 V)

# Constante de tiempo teórica tau = R * C
tau_teorico = R_nominal * C_nominal # 1.00 segundo
dVc_dt_inicial_teorico = V0_nominal / tau_teorico # 5.00 V/s

# ==========================================
# 2. MODELO DIFERENCIAL Y SOLUCIÓN ANALÍTICA
# ==========================================
def edo_circuito_rc(t, Vc, R, C, V0):
    """
    Ecuación Diferencial Ordinaria (EDO) de Carga RC:
    d(Vc)/dt = (V0 - Vc) / (R * C)
    """
    return (V0 - Vc) / (R * C)

def Vc_analitico_carga(t, V0, tau):
    """Solución analítica de la EDO para carga: Vc(t) = V0 * (1 - exp(-t / tau))"""
    return V0 * (1.0 - np.exp(-t / tau))

def dVc_dt_analitico_carga(t, V0, tau):
    """Derivada temporal de la carga: dVc/dt = (V0 / tau) * exp(-t / tau)"""
    return (V0 / tau) * np.exp(-t / tau)

# ==========================================
# 3. SIMULACIÓN NUMÉRICA Y GENERACIÓN DE DATOS
# ==========================================
# Dominio temporal t in [0, 5] s
t_eval = np.linspace(0.0, 5.0, 501)
dt = t_eval[1] - t_eval[0]

# Resolver EDO mediante método Runge-Kutta 4(5)
sol = solve_ivp(
    fun=edo_circuito_rc,
    t_span=(t_eval[0], t_eval[-1]),
    y0=[0.0],
    t_eval=t_eval,
    args=(R_nominal, C_nominal, V0_nominal),
    method='RK45'
)

Vc_simulado = sol.y[0]

# Adición de ruido blanco gaussiano N(0, sigma) para simular medición experimental
np.random.seed(42)  # Semilla para reproducibilidad
sigma_ruido = 0.02   # Desviación estándar del ruido en Voltios
Vc_experimental = Vc_simulado + np.random.normal(0, sigma_ruido, size=len(t_eval))

# Derivada numérica de los datos experimentales usando diferencias finitas centradas
dVc_dt_experimental = np.gradient(Vc_experimental, dt)

# ==========================================
# 4. AJUSTE DE CURVA POR MÍNIMOS CUADRADOS
# ==========================================
popt, pcov = curve_fit(Vc_analitico_carga, t_eval, Vc_experimental, p0=[5.0, 1.0])
V0_fit, tau_fit = popt
sigma_V0_fit, sigma_tau_fit = np.sqrt(np.diag(pcov))

# Cálculo del Coeficiente de Determinación (R^2)
residuos = Vc_experimental - Vc_analitico_carga(t_eval, *popt)
ss_res = np.sum(residuos**2)
ss_tot = np.sum((Vc_experimental - np.mean(Vc_experimental))**2)
r2_score = 1.0 - (ss_res / ss_tot)

# ==========================================
# 5. PROPAGACIÓN TEÓRICA DE ERRORES (GAUSS)
# ==========================================
# Incertidumbre en tau = R * C
sigma_tau_teorico = np.sqrt((C_nominal * sigma_R)**2 + (R_nominal * sigma_C)**2)

# Incertidumbre en (dVc/dt)_{t=0} = V0 / (R * C)
sigma_dV0_dt_teorico = dVc_dt_inicial_teorico * np.sqrt(
    (sigma_V0 / V0_nominal)**2 + (sigma_R / R_nominal)**2 + (sigma_C / C_nominal)**2
)

# ==========================================
# 6. IMPRESIÓN DE RESULTADOS EN CONSOLA
# ==========================================
print("==========================================================")
print(" RESULTADOS DEL ANÁLISIS DIFERENCIAL Y SIMULACIÓN DE CIRCUITO RC")
print("==========================================================")
print(f"Constante de tiempo teórica (tau):    {tau_teorico:.3f} ± {sigma_tau_teorico:.3f} s")
print(f"Constante de tiempo ajustada (fit):  {tau_fit:.3f} ± {sigma_tau_fit:.3f} s")
print(f"Tasa inicial de carga (dVc/dt)_{{t=0}}: {dVc_dt_inicial_teorico:.3f} ± {sigma_dV0_dt_teorico:.3f} V/s")
print(f"Voltaje asintótico (t -> inf):       {V0_fit:.3f} ± {sigma_V0_fit:.3f} V")
print(f"Coeficiente de Ajuste (R^2):          {r2_score:.5f}")
print("==========================================================")

# ==========================================
# 7. GRAFICACIÓN DE RESULTADOS
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

# --- Gráfica 1: Voltaje Vc(t) vs Tiempo ---
ax1.errorbar(t_eval[::20], Vc_experimental[::20], yerr=sigma_ruido, fmt='o', color='tab:blue', 
             ecolor='lightblue', elinewidth=1.5, capsize=3, label='Datos Experimentales (Simulados)')
ax1.plot(t_eval, Vc_analitico_carga(t_eval, V0_nominal, tau_teorico), 'r-', linewidth=2, 
         label=r'Ajuste Analítico: $V_c(t) = V_0(1 - e^{-t/\tau})$')
ax1.axhline(V0_nominal, color='gray', linestyle='--', label=r'Límite Asintótico $V_0 = 5.00\text{ V}$')
ax1.axvline(tau_teorico, color='green', linestyle=':', label=r'$\tau = 1.00\text{ s}$')
ax1.set_ylabel('Voltaje $V_c(t)$ [V]', fontsize=11)
ax1.set_title('Dinámica de Carga de un Condensador en un Circuito RC', fontsize=13, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right', fontsize=9)

# --- Gráfica 2: Derivada dVc/dt vs Tiempo ---
ax2.plot(t_eval, dVc_dt_experimental, '.', color='lightgray', alpha=0.7, label='Diferencias Finitas (Datos Ruidosos)')
ax2.plot(t_eval, dVc_dt_analitico_carga(t_eval, V0_nominal, tau_teorico), 'm-', linewidth=2, 
         label=r'Derivada Analítica: $\frac{dV_c}{dt} = \frac{V_0}{RC} e^{-t/RC}$')
ax2.axvline(tau_teorico, color='green', linestyle=':', label=r'$\tau = 1.00\text{ s}$')
ax2.set_xlabel('Tiempo $t$ [s]', fontsize=11)
ax2.set_ylabel(r'Tasa de Cambio $\frac{dV_c}{dt}$ [V/s]', fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right', fontsize=9)
# Análisis Diferencial y Simulación Numérica de Circuitos RC

**Asignatura:** Cálculo Diferencial  
**Institución:** Universidad de La Salle, Bogotá, Colombia  
**Autores:** 
* Diego Esteban Torres Puentes (`dtorres00@unisalle.edu.co`)
* Jhon Esteban Bedoya Vargas (`jvargas12@unisalle.edu.co`)

---

##  Descripción del Proyecto

Este repositorio contiene la implementación computacional del **Tema 18: Dinámica de Carga y Descarga de un Condensador en Circuitos RC**. El objetivo principal es analizar la tasa instantánea de cambio del voltaje $\frac{dV_c}{dt}$ mediante modelos analíticos de cálculo diferencial y simulación numérica orientada a la reproducibilidad.

---

##  Ecuaciones Principales

1. **Ecuación Diferencial Ordinaria (EDO):**
   $$RC \frac{dV_c(t)}{dt} + V_c(t) = V_0$$

2. **Solución del Voltaje de Carga:**
   $$V_c(t) = V_0 \left(1 - e^{-t/RC}\right)$$

3. **Derivada Temporal (Tasa Instantánea de Cambio):**
   $$\frac{dV_c(t)}{dt} = \frac{V_0}{RC} e^{-t/RC}$$

---

## Requisitos e Instalación

Para ejecutar este proyecto de forma local, asegúrate de tener Python 3.8+ instalado y ejecuta:

```bash
git clone [https://github.com/tu-usuario/calculo-diferencial-circuito-rc.git](https://github.com/tu-usuario/calculo-diferencial-circuito-rc.git)
cd calculo-diferencial-circuito-rc
pip install -r requirements.txt
python main.py
plt.tight_layout()
plt.savefig('grafica_circuito_rc.png', dpi=300)
plt.show()
