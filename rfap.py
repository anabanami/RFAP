# PHS3350
# Week 2 - wave packet and RFAP -
# Ana Fabela Hinojosa, 13/03/2021
import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import physunits
from scipy.fft import fft, ifft

plt.rcParams['figure.dpi'] = 200

folder = Path('wavepacket_time_evolution')
os.makedirs(folder, exist_ok=True)

# hbar = 1.0545718e-34 # [Js]
hbar = 1

𝜎 = 1
x_max = 10
x = np.linspace(-x_max,x_max, 1024)
n = x.size
x_step = 0.1

# oscillations per unit of space
k0 = 10 
# For Fourier space
k = 2 * np.pi * np.fft.fftfreq(n, x_step)

wave = np.exp(- x**2 / (2*𝜎**2)) * np.exp(1j*k0*x)

# Schrodinger equation (or first order time derivarive)
def Schrodinger_eqn(t, Ψ):
    return (-1j / hbar) * -hbar**2/(2 * m) * ifft((-k**2) * fft(Ψ)) + (-1j / hbar) * 1j*x**3 * Ψ

def Runge_Kutta(t, delta_t, Ψ):
    k1 = Schrodinger_eqn(t, Ψ)
    k2 = Schrodinger_eqn(t + delta_t / 2, Ψ + k1 * delta_t / 2) 
    k3 = Schrodinger_eqn(t + delta_t / 2, Ψ + k2 * delta_t / 2) 
    k4 = Schrodinger_eqn(t + delta_t, Ψ + k3 * delta_t / 2) 
    return Ψ + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


i = 0
t = 0
t_final = 1
delta_t = 0.01
m = 1
while t < t_final:

    plt.plot(x, np.real(wave), label="real part")
    plt.plot(x, np.imag(wave), label="imaginary part")
    plt.xlim(-x_max, x_max)
    plt.legend()
    plt.xlabel("x")
    plt.title(f"wave packet t = {i}")

    plt.savefig(folder/f'{i:04d}.png')
    # plt.show()
    plt.clf()
    
    wave = Runge_Kutta(t, delta_t, wave)
    i += 1
    t += delta_t

