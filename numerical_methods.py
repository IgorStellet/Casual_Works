import numpy as np


def Euler(f, xi, ti, tf, n, show=False):
    """
    Implementa o método de Euler para resolver equações diferenciais ordinárias.

    Parâmetros:
        f: função que calcula as derivadas (dx/dt)
        xi: valor inicial (pode ser escalar ou vetor)
        ti: tempo inicial
        tf: tempo final
        n: número de passos
        show: se True, mostra valores intermediários (opcional)

    Retorna:
        x: array com a solução em cada passo
        t: array com os tempos correspondentes
    """
    t = np.zeros(n)

    # Inicializa x apropriadamente para escalar ou vetor
    if isinstance(xi, (float, int)):
        x = np.zeros(n)
    else:
        x = np.zeros((n, len(xi)))

    # Condições iniciais
    x[0] = xi
    t[0] = ti
    h = (tf - ti) / n

    # Loop de integração
    for k in range(n - 1):
        if show:
            print(x[k], t[k])
        t[k + 1] = t[k] + h
        x[k + 1] = x[k] + h * f(x[k], t[k])

    return x, t

#------------------------------------------

def RungeKutta3(f, xi, ti, tf, n):
    """
    Implementa o método de Runge-Kutta de 3ª ordem para EDOs.

    Parâmetros:
        f: função que calcula as derivadas (dx/dt)
        xi: valor inicial (escalar ou vetor)
        ti: tempo inicial
        tf: tempo final
        n: número de passos

    Retorna:
        x: array com a solução
        t: array com os tempos
    """
    t = np.zeros(n)

    # Inicialização para escalar ou vetor
    if isinstance(xi, (float, int)):
        x = np.zeros(n)
    else:
        neq = len(xi)
        x = np.zeros((n, neq))

    # Condições iniciais
    x[0] = xi
    t[0] = ti
    dt = (tf - ti) / float(n)

    # Loop de integração
    for k in range(n - 1):
        K1 = dt * f(x[k], t[k])
        K2 = dt * f(x[k] + 0.5 * K1, t[k] + 0.5 * dt)
        K3 = dt * f(x[k] - K1 + 2 * K2, t[k] + dt)

        x[k + 1] = x[k] + (1 / 6.0) * (K1 + 4 * K2 + K3)
        t[k + 1] = t[k] + dt

    return x, t


#-------------------------------------

def RungeKutta4(f, xi, ti, tf, n):
    """
    Implementa o método de Runge-Kutta de 4ª ordem para EDOs.

    Parâmetros:
        f: função que calcula as derivadas (dx/dt)
        xi: valor inicial (escalar ou vetor)
        ti: tempo inicial
        tf: tempo final
        n: número de passos

    Retorna:
        x: array com a solução
        t: array com os tempos
    """
    t = np.zeros(n)

    # Inicialização para escalar ou vetor
    if isinstance(xi, (float, int)):
        x = np.zeros(n)
    else:
        neq = len(xi)
        x = np.zeros((n, neq))

    # Condições iniciais
    x[0] = xi
    t[0] = ti
    dt = (tf - ti) / float(n)
    dt2 = dt / 2.0

    # Loop de integração
    for k in range(n - 1):
        K1 = dt * f(x[k], t[k])
        K2 = dt * f(x[k] + 0.5 * K1, t[k] + dt2)
        K3 = dt * f(x[k] + 0.5 * K2, t[k] + dt2)
        K4 = dt * f(x[k] + K3, t[k] + dt)

        x[k + 1] = x[k] + (1 / 6.0) * (K1 + 2 * K2 + 2 * K3 + K4)
        t[k + 1] = t[k] + dt

    return x, t
