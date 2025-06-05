# Space Invaders – Dueling DQN with Reward Shaping  
**Kurs:** ITHS Deep Learning (HT23)

## Översikt
Detta projekt tränar en Dueling Double DQN agent för att spela *Space Invaders* med målet att inte bara förbättra agentens prestation, utan också förstå dess beslut. Genom att använda belöningsshaping (Reward Shaping) och förbättra utforskningen med NoisyDense-lager och Prioritized Experience Replay (PER), försöker vi skapa en agent som inte bara kan spela spelet effektivt utan också kan förklara sina handlingar.

## Huvudkomponenter
- **Dueling DQN**: En avancerad arkitektur som separerar beräkningen av Q-värden i två strömmar – värdet på ett tillstånd och fördelarna med specifika handlingar.
- **Reward Shaping**: Anpassade belöningar baserat på fiendens närhet och formationer för att främja strategiskt beteende.
- **Utforskningsförbättringar**: Användning av NoisyDense-lager och Prioritized Experience Replay för att uppmuntra agenten att prova olika strategier.
- **Utvärdering och loggning**: Visualisering och loggning av agentens beslut, policy och prestationer.

## Funktioner
- **Agentens Lärande**: Agenten lär sig genom trial-and-error och tränas med hjälp av belöningar för att uppmuntra strategiska handlingar.
- **Preprocessing**: Använder AtariPreprocessing och FrameStack för att förbehandla spelbilder till gråskala 84x84 pixlar och staplar fyra bilder för att ge agenten tidsmässigt sammanhang.
- **Replay Buffer & PER**: För effektiv inlärning används en Replay Buffer där agentens erfarenheter lagras och prioriteras baserat på dess förmåga att lära från dem.
- **Modellarkitektur**: Dueling DQN med NoisyDense-lager som gör det möjligt för agenten att utforska sina handlingar bättre.
- **Reward Shaping**: Speciellt utformade belöningsfunktioner för att förbättra agentens strategiska beslut.


## Installation

1. Klona repositoryn:
   ```bash
   git clone https://github.com/ditt-användarnamn/space-invaders-dueling-dqn.git
   cd space-invaders-dueling-dqn
   ````

2. Installera nödvändiga beroenden:

   ```bash
   pip install -r requirements.txt
   ```

3. Se till att du har rätt version av Python och bibliotek:

   * Python 3.7+
   * TensorFlow
   * Gymnasium (för Space Invaders-miljö)
   * Matplotlib och andra visualiseringsbibliotek

## Användning

### Träning

För att starta träningen, kör:

```bash
python trainer_main.py
```

### Utvärdering

För att utvärdera modellen kan du köra:

```bash
python evaluate_model.py
```

Du kan specificera olika parametrar för att köra utvärderingen, som vilka tränade modeller du vill utvärdera.

## Förklaring av de viktigaste filerna

* **trainer\_main.py**: Skriptet för att träna agenten.
* **trainer\_loop.py**: Innehåller själva träningsloopen och uppdateringar av modellen.
* **reward\_shaping.py**: Implementering av belöningsshaping-funktioner för att justera agentens beteende.
* **dueling\_dqn.py**: Modellarkitekturen för Dueling DQN som agenten använder.
* **prioritized\_replay.py**: Hantering av Replay Buffer och Prioritized Experience Replay.
* **video\_logger.py**: Loggar och sparar videofilmer av agentens spel.
* **utils/**: Hjälpfunktioner och verktyg som används i tränings- och utvärderingsprocessen.
