---
layout: post
title: Milestone 1
---

## Question 5 : Visualisations avancées

### 1. Heatmap interactives des saisons 2016 à 2020
Les 4 figures suivantes montrent les heatmap des positions des tirs de chaque équipe, pour les saisons 2016 à 2020. <br>
<u>Note</u> : les figures ci-dessous peut être regénérées en exécutant la dernière cellule 
du jupyter notebook './notebooks/Q5_Visualisations_Avancees.ipynb'
<a name="Q5-Season2016"></a> 
{% include season_2016_Regular.html %}

<a name="Q5-Season2017"></a> 
{% include season_2017_Regular.html %}

<a name="Q5-Season2018"></a> 
{% include season_2018_Regular.html %}

<a name="Q5-Season2019"></a>
{% include season_2019_Regular.html %}

<a name="Q5-Season2020"></a>
{% include season_2020_Regular.html %} 

### 2. Discussion générale
Les heatmap interactives permettent une comparaison de la position des tirs de chaque équipe, 
relativement à la moyenne de la ligue. D'une manière générale, on peut consigérer que davantage de 
tirs dans les zones dangeureuses (par exemple, devant les cages) entraîne davantage de buts. <br>

De façon plus générale, ces cartes permettent d'analyser les équipes sous différents angles :
<a name="Q21"></a>
 - Pour une équipe donnée, comment son jeu à évoluer sur les dernières années.
Est-ce qu'une évolution de la heatmap entraîne une remontée au classement ?
<a name="Q22"></a>
 - Pour une saison donnée, comment une équipe se compare aux autres équipe de la ligue.
 Est-ce qu'une "meilleure" heatmap entraîne un meilleur classement ?
<a name="Q23"></a>
 - Pour une équipe donnée et une saison donnée, quels ont été les zones d'où la plupart des tirs ont été pris.
Est-ce que les zones "chaudes" de la heatmap correspondent à des joueurs forts dans ces zones ?

### 3. Analyse de l'Avalanche Colorado
Les deux figures ci-dessous montrent les heatmap de l'Avalanche Colorado pour les saison 2016 (à gauche) 
et 2020 (à droite).<br>

<style>
figure{
    display: inline-block;
}
</style>

<figure>
    <img src='Colorado_2016.PNG' alt='Colorado_2016' width=250/>
    <figcaption>Shots map Colorado - 2016</figcaption>
</figure>
<figure>
    <img src='Colorado_2020.PNG' alt='Colorado_2020' width=250/>
    <figcaption>Shots map Colorado - 2020</figcaption>
</figure>

En analysant la [heatmap](#Q5-Season2016) de la saison 2016, on constate que l'Avalanche a tiré beaucoup moins devant les cages, et 
davantage loin des cages, comparées aux autres équipes de la ligue.
Ceci explique probablement leur [dernière place](https://www.nhl.com/standings/2016/league){:target="_blank"} au classement final.<br>

En analysant la [heatmap](#Q5-Season2020) de la saison 2020, on constate au contraire que l'Avalanche a tiré davantage que les 
autres équipes en face des cage et depuis le centre de la zone offensive. Ces zones sont les plus dangeureuses, 
et explique probablement leur [première place](https://www.nhl.com/standings/2020/league){:target="_blank"} au classement final. <br>

Comme supposé à la [section précédente](#Q21), il semblerait que pour une équipe donnée, 
une meilleure heatmap est corrélée avec un meilleur classement.

### 4. Comparaison des Sabres de Buffalo et du Lightning de Tampa Bay
Les deux figures ci-dessous montrent les heatmap deSabres de Buffalo (à gauche) et du 
Lightning de Tampa Bay (à droite) pour les saisons 2018, 2019 et 2020.<br>

<figure>
    <img src='Tampa_2018.PNG' alt='Tampa_2018' width=250/>
    <figcaption>Shots map Tampa - 2018</figcaption>
</figure>
<figure>
    <img src='Buffalo_2018.PNG' alt='Buffalo_2018' width=250/>
    <figcaption>Shots map Buffalo - 2018</figcaption>
</figure>

<figure>
    <img src='Tampa_2019.PNG' alt='Tampa_2019' width=250/>
    <figcaption>Shots map Tampa - 2019</figcaption>
</figure>
<figure>
    <img src='Buffalo_2019.PNG' alt='Buffalo_2019' width=250/>
    <figcaption>Shots map Buffalo - 2019</figcaption>
</figure>

<figure>
    <img src='Tampa_2020.PNG' alt='Tampa_2020' width=250/>
    <figcaption>Shots map Tampa - 2020</figcaption>
</figure>
<figure>
    <img src='Buffalo_2020.PNG' alt='Buffalo_2020' width=250/>
    <figcaption>Shots map Buffalo - 2020</figcaption>
</figure>

On constate nettement la capacité du Lightning à tirer davantage devant les cages et depuis 
le centre de la zone offensive, comparé aux autres équipes de la ligue. <br>
Au contraire, les Sabres ont une tendance à prendre leurs tirs loin des cages et moins 
depuis le centre de la zone offensive, comparé aux autres équipes de la ligue. 
Cette différence explique en partie pourquoi, sur les trois dernières années, 
[Tampay](https://en.wikipedia.org/wiki/Tampa_Bay_Lightning#Season-by-season_record){:target="_blank"} a mieux 
réussi que [Buffalo](https://en.wikipedia.org/wiki/Buffalo_Sabres#Season-by-season_record){:target="_blank"}.

Comme supposé à la [section précédente](#Q22), il semblerait que pour une saison donnée, 
une équipe ayant une meilleure heatmap aura un meilleur classement.<br>

Par ailleurs, on remarque pour les Sabres une nette tendance à prendre leurs tirs depuis la droite des cages. 
Cela semble être logique, étant donné que leur meilleur tireur est 
[Jack Eichel](https://en.wikipedia.org/wiki/Jack_Eichel){:target="_blank"} qui tire à droite.<br>

De plus, les tirs du Lightning sont plus équilibrés, car leurs meilleurs tireurs sont 
[Steven Stamkos](https://en.wikipedia.org/wiki/Steven_Stamkos){:target="_blank"} qui tire à droite 
et [Nikita Kucherov](https://en.wikipedia.org/wiki/Nikita_Kucherov){:target="_blank"} qui tire à gauche.<br>
 
Comme supposé à la [section précédente](#Q23), les zones "chaudes" de la heatmap correspondent 
donc à des joueurs forts dans ces zones.<br>

L'analyse des heatmap des tirs n'est évidemment pas suffisamment pour effectuer une analyse approfondie 
de la performance des équipes. Il faudrait également regarder : 
 - Le ratio tirs/buts pour savoir si une équipe est efficace.
 - Le nombre d'arrêt, pour savoir si un goal est performant.
 - La solidité de l'équipe en zone défensive.
 - La différence de niveau entre les différentes lignes d'attaque et de défense.
 - Et bien d'autres critères !

