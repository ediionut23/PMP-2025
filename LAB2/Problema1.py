import numpy as np

RED_INITIAL = 3
BLUE_INITIAL = 4
BLACK_INITIAL = 2

def simulate_experiment(n_simulations=100000):
    colors_drawn = []

    for _ in range(n_simulations):
        red = RED_INITIAL
        blue = BLUE_INITIAL
        black = BLACK_INITIAL

        die_roll = np.random.randint(1, 7)

        if die_roll in [2, 3, 5]:
            black += 1
        elif die_roll == 6:
            red += 1
        else:
            blue += 1

        urn = ['red'] * red + ['blue'] * blue + ['black'] * black
        ball_drawn = np.random.choice(urn)
        colors_drawn.append(ball_drawn)

    return colors_drawn

print("a) SIMULARE EXPERIMENT")
print("=" * 60)



n_simulations = 100000
colors_drawn = simulate_experiment(n_simulations)

red_count = colors_drawn.count('red')
blue_count = colors_drawn.count('blue')
black_count = colors_drawn.count('black')

print(f"simulări: {n_simulations}")
print(f"Bile roșii extrase: {red_count}")
print(f"Bile albastre extrase: {blue_count}")
print(f"Bile negre extrase: {black_count}")

print("\nb) PROBABILITATE ESTIMATĂ ROȘIE")
print(f"P(roșu) = {red_count/n_simulations:.6f}")

print("\nc")

p_prime = 3/6
p_six = 1/6
p_other = 2/6
#se calculeaza probabilitaea de a extrage pentru fiecare culoare in cazul in care pe zar e prim, sase sau alta valoare
#apoi se face media ponderata in functie de probabilitatea de a cadea fiecare valoare pe zar
#astfel se obtine probabilitatea totala de a extrage fiecare culoare
#se observa ca valorile sunt foarte apropiate de cele obtinute prin simulare
p_red = p_prime * (RED_INITIAL/10) + p_six * ((RED_INITIAL+1)/10) + p_other * (RED_INITIAL/10)
p_blue = p_prime * (BLUE_INITIAL/10) + p_six * (BLUE_INITIAL/10) + p_other * ((BLUE_INITIAL+1)/10)
p_black = p_prime * ((BLACK_INITIAL+1)/10) + p_six * (BLACK_INITIAL/10) + p_other * (BLACK_INITIAL/10)

print(f"P(roșu)     = {p_red:.6f}")
print(f"P(albastru) = {p_blue:.6f}")
print(f"P(negru)    = {p_black:.6f}")

print("\nCOMPARAȚIE:")
print(f"Diferență roșu:     {abs(red_count/n_simulations - p_red):.6f}")
print(f"Diferență albastru: {abs(blue_count/n_simulations - p_blue):.6f}")
print(f"Diferență negru:    {abs(black_count/n_simulations - p_black):.6f}")
