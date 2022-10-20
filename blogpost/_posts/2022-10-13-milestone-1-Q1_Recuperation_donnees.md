---
layout: post
title: Milestone 1
---

## Question 1 : Acquisition des données

Les lignes qui vont suivre sont un bref tutoriel permettant de télécharger les données 
"play-by-play" des matchs de NHL à l'aide de notre code, disponible sur GitHub : <br>
[An Ni Wu](https://github.com/anw1998/IFT6758-A2022-G08) <br>
[Thomas Rives](https://github.com/THOMAS921) <br>
[Rui Ze Ma](https://github.com/ruizema) <br>
[Alain Fidahoussen](https://github.com/AlainFidahoussen/IFT6758-A2022-G08.git) <br>

A la fin de ce tutoriel, vous serez en mesure d'obtenir toutes les 
informations officielles rendues disponibles par l'API des statistiques de la NHL 
pour les saisons que vous désirez.

### Étape 1 : Récupération du code

Dans un premier temps, il vous sera nécessaire de récupérer notre code en clonant un des répertoires Git 
ci-dessus. Pour cela, sur votre terminal, placez-vous à l'endroit où vous souhaitez récupérer le code et exécutez, par exemple, la commande suivante:

```console
git clone https://github.com/anw1998/IFT6758-A2022-G08
```
Cette commande une fois exécutée, vous devriez avoir téléchargé un dossier nommé **IFT6758-A2022-G08-master**. Le code permettant de télécharger les jeux de données s'y trouve. 

### Étape 2 : Définition de la variable d'environnement
Pour éviter de multiples requêtes via une API publique, les données sont téléchargées dans un répertoire local que 
vous devez définir soit en tant qu'argument du constructeur (cf. étape 3.), soit par la variable d'environnement *NHL_DATA_DIR*.
Notez que ce répertoire sera automatiquement créé si non existant.

### Étape 3 : Importation du module et création du data manager

Depuis le dossier **IFT6758-A2022-G08-master**, lancez l'interpréteur Python (ou créez un 
fichier .py), importez le module *src.data.NHLDataManager* et instanciez un objet de type *NHLDataManager* :

```python
import src.data.NHLDataManager as DataManager

# Les données seront téléchargées dans 'directory_name'
data_manager = DataManager.NHLDataManager('directory_name')
# OU
# Les données seront téléchargées dans la répertoire définit par la 
# variable d'environnement NHL_DATA_DIR
data_manager = DataManager.NHLDataManager()
```

### Étape 4: Téléchargement des données

Pour le moment, seul le téléchargement des saisons régulières et des playoffs est supporté.
Le téléchargement se fait sous la forme d'un fichier [json](https://en.wikipedia.org/wiki/JSON) par match, dont le nom est définit 
par un identificateur unique ([Game IDs](https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md#game-ids))

**La façon recommandée** pour récupérer les données des saisons 2016 à 2020 est de lancer la commande :
```python
make data
```
Les données brutes se trouveront alors dans le sous-répertoire 'raw'. 
Il est à noter que d'autres données (dataframe, csv) seront calculées et sauvergardées dans le sous-répertoire 'processed'. <br>

Il est également possible de faire directement appel à certaines fonctions pour :

 - Le téléchargement sur disque (json) d'une ou plusieurs saisons : 

```python
# Liste des saisons à télécharger
seasons_year = [2016, 2017, 2018]

# Type de saison (régulière)
season_type = "Regular"
data_manager.download_data(
   seasons_year=seasons_year, 
   season_type=season_type)

# Type de saison (playoffs)
season_type = "Playoffs"
data_manager.download_data(
   seasons_year=seasons_year, 
   season_type=season_type)
```

 - Le téléchargement sur disque (json) et en mémoire (dictionaire python) d'une saison entière :

```python
# Saison à télécharger
season_year = 2016

# Type de saison
season_type = "Regular"

data_season = data_manager.load_data(
   season_year=season_year, 
   season_type=season_type)
```

 - Le téléchargement sur disque (json) et en mémoire (dictionaire python) d'un match précis :

```python
# Saison à télécharger
season_year = 2016

# Type de saison
season_type = "Regular"

# Numéro du match
game_number = 12

data_game = data_manager.load_game(
   season_year=season_year, 
   season_type=season_type,
   game_number=game_number)
```

Il ne suffit alors à présent plus qu'à vérifier que le téléchargement s'est correctement effectué. Par exemple : 
```python
data_raw_dir = os.path.join(os.environ['NHL_DATA_DIR'], 'raw')
season_year = 2016,
season_type 'Regular'
path_data = os.path.join(data_dir, str(season_year), season_type)

json_files = os.listdir(path_data)

print(json_files[0:3])
['2016020001.json', '2016020002.json', '2016020003.json']

print(len(json_files))
1231
```

## Question 2


