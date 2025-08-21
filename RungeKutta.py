import numpy as np
import matplotlib.pyplot as plt

# Função Principal para aplicar o método de Euler


def Euler(f, ri, ti, tf, n, show=False):
    """
    Metodo de euler para o vetor r'=f(x,t)
    :param f: Função
    :param ri: Valor inicial de do vetor r  composto por diversos parâmetros
    :param ti: Tempo inicial (Variavel indepedente)
    :param tf: Tempo final   (idem)
    :param n: Quantidade de Passos
    :param show: Parametro opicional para printar os valores de x,y...nos determinados tempo t
    :return: Lista dos valores de r e t obtidos com o método de euler
    """
    t = np.zeros(n)
    # Condicional para o programa funcionar indepedente de ser mais de uma variável ou não
    if isinstance(ri, (float, int)):
        r = np.zeros(n)
    else:
        r = np.zeros((n, len(ri)))
    r[0] = ri
    t[0] = ti
    h = (tf - ti) / n
    for k in range(n - 1):
        if show:
            print(r[k, 0], r[k, 1], r[k, 2], r[k, 3], t[k])
        t[k + 1] = t[k] + h
        r[k + 1] = r[k] + h * f(r[k], t[k])
    return r, t

# -------------------------------
# Constantes que serão utilizadas


Msol = 1.98E30
G = 6.67E-11


# Função que colocamos a derivada que utilizamos, que no no caso está na ordem r'=[x', x'', y', y'']
# Com x' e y' sendo a velocidade que é dado pelas componentes do vetor r, ou seja, r[1] e r[3], respectivamente.


def derivada(r, t):
    """Derivada das funções que desejamos aplicar o método de euler"""
    return np.array([r[1], -G * Msol * r[0] / ((r[0] ** 2 + r[2] ** 2) ** (3 / 2)), r[3],
                     -G * Msol * r[2] / ((r[0] ** 2 + r[2] ** 2) ** (3 / 2))])




# ---------------------------------------
# Parte do programa para definirmos nossos valores iniciais/ finais e passos


xi = np.array([1.496E11, 0, 0, 2.97E4])
ti = 0
tf = 365 * 24 * 60 * 60
dia = 365  # Passo para o dia
hora = 365 * 24  # Passo para a hora
minuto = 365 * 24 * 60  # Passo para o minuto


ResDia, t1 = Euler(derivada, xi, ti, tf, dia)
ResHr, t2 = Euler(derivada, xi, ti, tf, hora)
ResMinuto, t3 = Euler(derivada, xi, ti, tf, minuto)


# -------------------------------------
# Parte do programa para plotarmos o gráfico


# Gráfico para passos de 1 dia
#plt.style.use('seaborn')
plt.axes().set_aspect('equal')
plt.title(r"Trajetória da terra - Passos de 1 dia")
plt.ylabel(r'Posição vertical')
plt.grid(True)
plt.xlabel("Posição Horizontal")
plt.plot(ResDia[:, 0], ResDia[:, 2], 'lightblue')
plt.legend(['Trajetória da terra Euler - Passos de 1 dia'], loc='lower left')
plt.savefig('kepler_dia.png')
plt.show()


# Gráfico para passos de 1 hora
#plt.style.use('seaborn')
plt.axes().set_aspect('equal')
plt.title(r"Trajetória da terra - Passos de 1 hora")
plt.ylabel(r'Posição vertical')
plt.grid(True)
plt.xlabel("Posição Horizontal")
plt.plot(ResHr[:, 0], ResHr[:, 2], 'violet')
plt.legend(['Trajetória da terra Euler - Passos de 1 hora'], loc='lower left')
plt.savefig('kepler_hora.png')
plt.show()


# Gráfico para passos de 1 minuto
#plt.style.use('seaborn')
plt.axes().set_aspect('equal')
plt.title(r"Trajetória da terra - Passos de 1 minuto")
plt.ylabel(r'Posição vertical')
plt.grid(True)
plt.xlabel("Posição Horizontal")
plt.plot(ResMinuto[:, 0], ResMinuto[:, 2], 'gold')
plt.legend(['Trajetória da terra Euler - Passos de 1 minuto'], loc='lower left')
plt.savefig('kepler_minuto.png')
plt.show()


"""Podemos concluir dos resultados que o método de euler pode ser realmente muito
eficiente e prático uma vez que o programa já está montado e sabe-se quais são as derivadas das equações
que desejamos visualizar as aproximações. Também é possível verificar que quanto maior a quantidade de passos,
ou seja, quanto menor o valor de h mais preciso se torna o gráfico da equação que estamos tratando."""


# --------------------------------
# Tarefa 7, substituindo a chamada do método de Euler para o método de RungeKutta


from rk4 import RungeKutta4


x, t = RungeKutta4(derivada, xi, ti, tf, dia)


# Gráfico para passos de 1 dia no método RungeKutta
#plt.style.use('seaborn')
plt.axes().set_aspect('equal')
plt.title(r"Trajetória da terra - Passos de 1 dia")
plt.ylabel(r'Posição vertical')
plt.grid(True)
plt.xlabel("Posição Horizontal")
plt.plot(x[:, 0], x[:, 2], 'lightblue')
plt.legend(['Trajetória da terra RungeKutta - Passos de 1 dia'], loc='lower left')
plt.savefig('RungeKutta_dia.png')
plt.show()


# Podemos observar o quão mais eficiente é esse método comparando os gráficos de 1 dia obtido
# Com esse método o gráfico de 1 dia praticamente fechou a órbita enquanto o método anterior não chegou perto.
