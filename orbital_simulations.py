import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import Euler, RungeKutta3, RungeKutta4

# ==============================================
# Constantes Físicas e Função Derivada
# ==============================================

# Constantes astronômicas
Msol = 1.98E30  # Massa do Sol (kg)
G = 6.67E-11  # Constante gravitacional (m^3 kg^-1 s^-2)


def derivada(r, t):
    """
    Calcula as derivadas para o movimento orbital em 2D.
    O vetor de estado r contém: [x, vx, y, vy]

    Retorna:
        Array com as derivadas [vx, ax, vy, ay] - Com v sendo a velocidade e a a aceleração
    """
    r_norm = (r[0] ** 2 + r[2] ** 2) ** (3 / 2)
    ax = -G * Msol * r[0] / r_norm
    ay = -G * Msol * r[2] / r_norm

    return np.array([r[1], ax, r[3], ay])


# ==============================================
# Configuração da Simulação
# ==============================================

# Condições iniciais (posição e velocidade da Terra)
xi = np.array([1.496E11, 0, 0, 2.97E4])  # [x, vx, y, vy] em m e m/s

# Parâmetros temporais
ti = 0  # Tempo inicial (s)
tf = 365 * 24 * 60 * 60  # 1 ano em segundos

# Opções de discretização temporal (n, quantidade de passos)
dia = 365  # Passos de 1 dia
hora = 365 * 24  # Passos de 1 hora
minuto = 365 * 24 * 60  # Passos de 1 minuto


# ==============================================
# Função Auxiliar para Plotagem
# ==============================================

def plot_orbita(dados, cor, titulo, nome_arquivo):
    """Função auxiliar para plotar órbitas."""
    plt.figure()
    plt.axes().set_aspect('equal')
    plt.title(titulo)
    plt.ylabel('Posição vertical (m)')
    plt.xlabel('Posição horizontal (m)')
    plt.grid(True)
    plt.plot(dados[:, 0], dados[:, 2], color=cor)
    plt.legend([titulo], loc='lower left')
    plt.savefig(nome_arquivo)
    plt.show()


# ==============================================
# Simulações com Diferentes Métodos
# ==============================================

# Método de Euler
ResEulerDia, _ = Euler(derivada, xi, ti, tf, dia)
ResEulerHr, _ = Euler(derivada, xi, ti, tf, hora)
ResEulerMinuto, _ = Euler(derivada, xi, ti, tf, minuto)

# Método de Runge-Kutta 3ª ordem
ResRK3Dia, _ = RungeKutta3(derivada, xi, ti, tf, dia)

# Método de Runge-Kutta 4ª ordem
ResRK4Dia, _ = RungeKutta4(derivada, xi, ti, tf, dia)

# ==============================================
# Visualização dos Resultados
# ==============================================

# Resultados com Euler
plot_orbita(ResEulerDia, 'lightblue',
            "Trajetória da Terra (Euler) - Passos de 1 dia",
            'euler_dia.png')

plot_orbita(ResEulerHr, 'violet',
            "Trajetória da Terra (Euler) - Passos de 1 hora",
            'euler_hora.png')

plot_orbita(ResEulerMinuto, 'gold',
            "Trajetória da Terra (Euler) - Passos de 1 minuto",
            'euler_minuto.png')

# Resultados com Runge-Kutta
plot_orbita(ResRK3Dia, 'lightgreen',
            "Trajetória da Terra (RK3) - Passos de 1 dia",
            'rk3_dia.png')

plot_orbita(ResRK4Dia, 'salmon',
            "Trajetória da Terra (RK4) - Passos de 1 dia",
            'rk4_dia.png')

# ==============================================
# Análise Comparativa
# ==============================================

"""
Observações:
1. O método de Euler requer passos temporais muito pequenos para boa precisão.
2. Runge-Kutta 3ª ordem já mostra melhoria significativa com o mesmo passo temporal.
3. Runge-Kutta 4ª ordem fornece os melhores resultados com passos grandes.
4. Para problemas orbitais, métodos de ordem superior são mais eficientes.
"""


## 6. Análise de Energia (Bônus)
#Vamos verificar a conservação de energia nos diferentes métodos.

def calculate_energy(r):
    """Calcula energia mecânica total (cinética + potencial)"""
    kinetic = 0.5 * (r[1]**2 + r[3]**2)
    potential = -G * Msol / np.sqrt(r[0]**2 + r[2]**2)
    return kinetic + potential

# Simulação com mais passos para análise suave
steps = 1000
euler_res, t_euler = Euler(derivada, xi, ti, tf, steps)
rk4_res, t_rk4 = RungeKutta4(derivada, xi, ti, tf, steps)


# Calcula energia ao longo do tempo
energy_euler = np.array([calculate_energy(r) for r in euler_res])
energy_rk4 = np.array([calculate_energy(r) for r in rk4_res])

# Plota energia
plt.figure(figsize=(10, 6))
plt.plot(t_euler, energy_euler, label='Euler')
plt.plot(t_rk4, energy_rk4, label='RK4')
plt.title('Energia Mecânica Total ao Longo do Tempo')
plt.xlabel('Tempo (s)')
plt.ylabel('Energia (J/kg)')
plt.legend()
plt.grid(True)
plt.show()
