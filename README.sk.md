# edge-or-noise

[![tests](https://github.com/nadeko0/edge-or-noise/actions/workflows/tests.yml/badge.svg)](https://github.com/nadeko0/edge-or-noise/actions/workflows/tests.yml)

*[English](README.md) · [Русский](README.ru.md) · Slovenčina*

Výskumný projekt: dokáže ML klasifikátor nájsť obchodovateľnú výhodu
na krátkom horizonte v **tikových** dátach z perpetuálnych futures na
Bybit (obchody, order book, likvidácie, funding rate, open interest) —
na rozdiel od pomalších sviečkových/indikátorových stratégií, ktoré tu
nie sú predmetom skúmania?

**Tento repozitár je výskumný denník, nie obchodný systém.**
Dokumentuje metodológiu a jej výsledky vrátane negatívnych. Nič v ňom
nie je odporúčaním obchodovať so skutočným kapitálom.

**Čo tento projekt demonštruje, ak repozitár prezeráte kvôli
zručnostiam a nie výsledkom:** asynchrónny WebSocket kolektor dát
(`asyncio`/`websockets`, reconnect/backoff, denne rotované komprimované
úložisko); dátový pipeline (Polars/Pandas/NumPy/Numba) spracúvajúci
desiatky miliónov riadkov na symbol; prísna ML validácia
(chronologický walk-forward s purge medzerou, permutačné testovanie,
potvrdenie na nezávislom roku dát mimo vzorky); štatistické povedomie
o falošne pozitívnych výsledkoch namiesto publikovania prvého pekne
vyzerajúceho backtestu; a modulárny, testovaný (`pytest`), so
zamknutými závislosťami (`uv`) kód, kde je zber dát oddelený od
výskumného kódu.

---

## Prečo tento projekt existuje

Hľadal som výhodu pre scalping na krátkom horizonte v tikových dátach,
pretože som chcel veriť, že order flow a hĺbka order booku obsahujú
niečo, čo pomalšia, sviečková stratégia nevidí. Neobsahujú — aspoň nie
v týchto dátach, nie po zhruba 80 čestne otestovaných hypotézach.
Radšej to zverejním priamo, než by som ticho pochoval negatívny
výsledok a ukázal len tie časti, ktoré vyzerajú dobre.

V tomto repozitári je skutočná stopa: chyba o jeden (off-by-one),
ktorú som našiel a opravil vo vlastnom validačnom kóde, konfigurácia,
ktorá vyzerala vynikajúco na 17 obchodoch a rozpadla sa na 2 000, štyri
modelové enginy, ktoré sa všetky navzájom zhodli (čo je tiež svojím
spôsobom odpoveď). Ak toto posudzujete pre pracovnú pozíciu, pýtajte sa
ma na čokoľvek priamo — spustil som každý experiment tu, rozumiem,
prečo každý z nich zlyhal alebo obstál, a radšej detailne obhájim
negatívny výsledok, než by som prikrášlil falošne pozitívny.

Pre presnosť, keďže je ľahké toto zaokrúhliť: táto konkrétna, prísna
práca s walk-forward validáciou má približne **7 mesiacov** (od
približne decembra 2025). Predtým, od mája 2025, boli neformálne
skúšobné behy na spotovom obchodovaní so základnými indikátormi cez
TA-Lib — experimentálne, nie na úrovni prísnosti ukázanej v tomto
repozitári. Radšej to poviem otvorene, než aby to znelo, že táto
úroveň prísnosti existovala celý rok, keď to tak nebolo.

Časť mojej skoršej práce na sviečkových (nie tikových) dátach tiež
priniesla backtesty, ktoré vyzerali pôsobivo — profit factor nad 2,
takmer žiadny drawdown — bez permutačného testu alebo skutočného
potvrdenia mimo vzorky za tým. Takémuto výsledku už sám o sebe
neverím — presne preto je tento repozitár postavený tak, ako je:
každé číslo tu musí prejsť rovnakými tromi kontrolami (či prežije
nulové rozdelenie na premiešaných štítkoch, či drží na dátach, ktoré
nikdy nevidel, či prežije mierne zmenené parametre) — až potom niečo
znamená.

---

## Hlavný výsledok

Bol nájdený štatisticky reálny prediktívny signál, konzistentný naprieč
aktívami (cross-asset) aj naprieč rokmi (cross-year) —
**ROC-AUC ≈ 0,58–0,67** v závislosti od horizontu predikcie (AUC meria
schopnosť radenia: 0,5 = nie lepšie ako hod mincou, 1,0 = dokonalé; ide
o reálnu, no skromnú výhodu, nie silnú). Je takmer úplne
vysvetlený krátkodobým zhlukovaním volatility, nie order flow, hĺbkou
order booku ani smerom. Žiadna z testovaných konfigurácií (~80
hypotéz: varianty stop-lossu, 4 modelové enginy, hĺbka order booku,
technické indikátory, páka, funding rate, open interest a niekoľko
doslovne zakódovaných konceptov "smart money") nedosiahla **profit
factor** spoľahlivo nad 1,0 (profit factor = súčet ziskových $ /
súčet stratových $; nad 1,0 znamená čistý zisk po poplatkoch) na
vzorke dostatočne veľkej na to, aby sa jej dalo veriť.
Kompletné čísla: [`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md)
(v angličtine).

**Forward test, 2026-10-18.** Kolektor bežal ďalej, takže pôvodný model
bol zmrazený a raz vyhodnotený na 89 neskorších dňoch
(2026-07-21 → 2026-10-17), na ktorých sa v tomto repozitári nič
neladilo. Kritériá úspechu a neúspechu boli stanovené pred spustením
([`experiments/09_forward_test.py`](experiments/09_forward_test.py)).
Všetko platí: AUC 0,640 pri 5-minútovom horizonte (maximum pri
premiešaných označeniach 0,541), samotná realizovaná volatilita dáva 99 %
z toho, model BTC na ETH 0,591 / SOL 0,577 a žiadne z 9 prahových
pravidiel nedosiahlo PF > 1 na vzorke 200+ obchodov (osem buniek s PF
0,64–0,82; jediná nad 1 má 10 obchodov, rovnaká pasca malej vzorky ako
predtým). Pri tom sa našla a opravila chyba v čítaní order booku
(pozri nižšie).
Podrobnosti: [`FINAL_REPORT.md` §10](reports/FINAL_REPORT.md#10-forward-test-2026-10-18).

<p align="center">
  <img src="figures/forward_test.png" width="85%" alt="Zoskupený stĺpcový graf: ROC-AUC podľa horizontu predikcie, pôvodný walk-forward oproti zmrazenému modelu na 89 nevidených dňoch; forward stĺpce sú vyššie pri každom horizonte, od 0,722 pri 1 minúte po 0,619 pri 10 minútach.">
</p>

<p align="center">
  <img src="figures/permutation_test.png" width="85%" alt="Pozorované AUC vynesené oproti nulovému rozdeleniu z 60 behov s premiešanými štítkami — reálny výsledok leží ďaleko mimo miesta, kam sa dostávajú náhodné štítky.">
</p>
<p align="center">
  <img src="figures/auc_by_horizon.png" width="85%" alt="Stĺpcový graf: ROC-AUC klesá z 0,669 pri 1-minútovom horizonte na 0,580 pri 10-minútovom horizonte.">
</p>
<p align="center">
  <img src="figures/feature_importance.png" width="85%" alt="Horizontálny stĺpcový graf dôležitosti príznakov XGBoost: dominuje kĺzavá realizovaná volatilita a rozsah baru; príznaky order flow sú takmer na dne.">
</p>

---

## Dáta

| Zdroj | Kto zbieral | Zbierané symboly | Obdobie | Súčasť repozitára? |
|---|---|---|---|---|
| Živé tikové dáta (obchody, order book, likvidácie, tickers/funding/OI) | Vlastný WS kolektor tohto projektu (`collector/`) | 24 perpetuálnych kontraktov Bybit (USDT) | 2026-04-22 → 2026-07-20 (90 dní, pôvodná štúdia) | Nie — surové dáta majú ~100 GB, sú v .gitignore |
| Ten istý kolektor, neskoršie dáta (iba forward test) | ten istý | tie isté (26 približne od 2026-09-30) | 2026-07-21 → 2026-10-17 (89 dní) | Nie |
| Historické tikové dáta (kontrola mimo vzorky) | Archív tretej strany nájdený online (nezbieral tento projekt) | BTC, ETH, SOL | 2024-02-12 → 2024-06-02 (112 dní) | Nie — nie je možné ho ďalej šíriť |

Kolektor zbiera 24 symbolov, ale skutočné experimenty v tomto
repozitári používajú iba **BTCUSDT** (hlavný, ~80 hypotéz), plus
**ETHUSDT** a **SOLUSDT** pre cross-asset kontroly na rovnakom
90-dňovom živom okne. Historický archív pokrýva BTC/ETH/SOL, ale
kontrola mimo vzorky v tomto repozitári (`experiments/06`) trénuje a
vyhodnocuje iba na jeho **BTC** dátach — presné rozdelenie
symbol/obdobie podľa hypotézy nájdete v
[`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md#data-coverage-by-hypothesis)
(v angličtine), vrátane zmeny formátu dát kolektora uprostred datasetu
(2026-05-01/02).

**Prečo nie všetkých 24:** nezávislý archív z roku 2024 — najprísnejšia
kontrola v tomto repozitári — pokrýva iba BTC/ETH/SOL, takže rozšírenie
batérie ~80 hypotéz na zvyšných 21 symbolov by pripravilo analýzu o
jediný test, ktorý naozaj odlíši reálny vzor od falošne pozitívneho
výsledku. ETH/SOL boli vybrané na overenie, či sa zistenie z BTC
prenáša na iné likvidné, etablované mince, nie na skenovanie celého
zoznamu symbolov kvôli samostatnej výhode; spustenie rovnakej batérie
pre každý symbol tiež lineárne rastie výpočtovo (samotný replay L2
order booku trvá ~15 minút na symbol). Preskenovanie zvyšných altcoinov
— ktoré sú tenšie a hlučnejšie a vyžadovali by vlastnú opatrnosť — je
prirodzeným rozšírením, nie niečím, čo tento repozitár tvrdí, že
vylúčil.

Oba datasety zdieľajú rovnaký pipeline na tvorbu barov a príznakov
(`tickml/loaders_live.py`, `tickml/loaders_sferez.py`) — práve to robí
zmysluplným porovnanie mimo vzorky v
`experiments/06_sferez_oos_confirmation.py`: rovnaké príznaky, rovnaké
označovanie, rovnaký kód, ale iný rok, ktorý žiadny model v tomto
repozitári nevidel pri trénovaní.

Na reprodukovanie čohokoľvek tu potrebujete vlastnú kópiu podobne
štruktúrovaných tikových dát. Nastavte:

```bash
export TICKML_DATA_DIR=/path/to/your/collector/output
export TICKML_SFEREZ_DIR=/path/to/your/2024/historical/archive   # voliteľné
```

---

## Architektúra

```mermaid
flowchart TD
    subgraph Collection["Zber dát"]
        direction TB
        WS["Verejný WS Bybit<br/>(obchody, likvidácie,<br/>tickers, orderbook.50,<br/>orderbook.rpi)"]
        COL["collector/<br/>(asynchrónny WS klient,<br/>reconnect + backoff)"]
        DISK["data/*.jsonl.gz<br/>(jeden súbor na symbol a deň)"]
        WS --> COL --> DISK
    end

    subgraph Research["tickml/ výskumný pipeline"]
        direction TB
        LOAD["loaders_live.py /<br/>loaders_sferez.py<br/>(gzip -> 1-min bary)"]
        L2["orderbook_l2.py<br/>(snímky L2 order booku,<br/>detekcia whale-wall)"]
        FEAT["features.py<br/>(príznaky založené len<br/>na minulých dátach)"]
        LABEL["labeling.py<br/>(triple-barrier<br/>PT / SL / time-exit)"]
        MODEL["models.py<br/>(XGBoost / LightGBM /<br/>CatBoost / RandomForest)"]
        VALID["validation.py<br/>(walk-forward, purge,<br/>permutačný test, OOS,<br/>equity curve)"]
        LOAD --> FEAT
        L2 --> FEAT
        FEAT --> LABEL --> MODEL --> VALID
    end

    Collection --> Research
    VALID --> REPORT["reports/FINAL_REPORT.md"]
```

---

## Metodológia

### 1. Triple-barrier označovanie

Každý 1-minútový bar je označený nezávisle od akéhokoľvek obchodného
pravidla: "keby obchod vstúpil na otvorení nasledujúceho baru, bol by
ziskový?" Toto oddeľuje otázku "dokáže model zoradiť bary podľa
budúceho výsledku" od otázky "funguje ručne vybrané pravidlo" — druhé
je to, čo v skutočnosti testuje väčšina verejných scalping stratégií,
čím tieto dve otázky zamieňa.

```mermaid
flowchart TD
    A["Bar i sa uzavrel"] --> B["Hypotetický vstup na<br/>otvorení baru i+1"]
    B --> C{"Počas nasledujúcich<br/>max_bars barov..."}
    C -->|"cena dosiahla<br/>take-profit"| D["label = 1<br/>pnl = +PT - poplatky"]
    C -->|"cena dosiahla<br/>stop-loss (voliteľné)"| E["label = 0<br/>pnl = -SL - poplatky"]
    C -->|"ani jedno<br/>za max_bars barov"| F["výstup na close<br/>posledného baru (time-exit)<br/>label = pnl > 0"]
```

### 2. Walk-forward validácia s purge medzerou

```mermaid
flowchart LR
    subgraph Fold1["Fold 1"]
        T1["train: 0 % – 55 %"] --> G1["purge<br/>(max_bars+1 riadkov)"] --> V1["test: 55 % – 70 %"]
    end
    subgraph Fold2["Fold 2"]
        T2["train: 0 % – 70 %"] --> G2["purge"] --> V2["test: 70 % – 85 %"]
    end
    subgraph Fold3["Fold 3"]
        T3["train: 0 % – 85 %"] --> G3["purge"] --> V3["test: 85 % – 100 %"]
    end
    Fold1 --> Fold2 --> Fold3
```

Purge medzera existuje preto, že štítok na bare *i* závisí od výsledkov
až do `max_bars` barov v budúcnosti — bez nej by tréningový riadok
blízko hranice foldu mohol mať štítok, ktorý "vidí" do testovacieho
okna.

### 3. Kontrola na nezávislom datasete mimo vzorky

Najsilnejší test v tomto repozitári: natrénovať model raz na celom
živom (2026) datasete, potom ho vyhodnotiť na historickom datasete z
roku 2024, ktorý nebol nikdy použitý pri trénovaní, výbere príznakov
ani výbere prahov. Vzor, ktorý existuje len v okne, na ktorom bol
nájdený, tento test neprežije;
`experiments/06_sferez_oos_confirmation.py` ukazuje jednak prípad, kde
hlavné AUC zistenie *obstojí*, aj prípad, kde dva "výborné" výsledky na
malej vzorke (n=17–26 obchodov) *neobstoja* — druhý sa vráti späť k
nule alebo pod ňu po opätovnom otestovaní na tisícoch nezávislých
obchodov, čo je učebnicový príznak falošne pozitívneho výsledku
spôsobeného mnohonásobným porovnávaním, nie reálneho vzoru.

---

## Čo bolo otestované a zamietnuté

Plne zdokumentované s číslami v
[`reports/FINAL_REPORT.md`](reports/FINAL_REPORT.md) (v angličtine).
Zhrnutie:

- Varianty stop-lossu (pevné aj škálované podľa volatility) — žiadne
  zlepšenie, v niekoľkých prípadoch horšie než bez stop-lossu vôbec.
- Smer obchodu (long vs. short) — štatisticky identické, dôkaz, že
  signál nie je smerový.
- Hĺbka order booku nad rámec top-of-book, zo snímok order booku s 50
  úrovňami, vrátane explicitného príznaku "whale wall" (neprimerane
  veľká jednotlivá objednávka) — žiadny merateľný efekt, potvrdené
  na celom 90-dňovom okne aj na obmedzenom okne s jednotnou schémou
  (pozri nižšie). Oprava (2026-10): pôvodná verzia čítala tieto snímky
  ako delty, čo viedlo k neplatným príznakom hĺbky (order book
  prekrížený v každej vzorke). Po oprave je výsledok rovnaký, na
  pôvodnom okne aj na 89-dňovom forward okne
  ([report §2, §4](reports/FINAL_REPORT.md#2-bug-audit)).
- 4 modelové enginy (XGBoost, LightGBM, CatBoost, RandomForest) —
  zameniteľné výsledky.
- Kompletná klasická knižnica technických indikátorov (RSI, MACD,
  Bollinger, ADX, Stochastic, SuperTrend, Hurstov exponent, CMF,
  Fisherova transformácia, KAMA, MFI, sklon/R² lineárnej regresie,
  z-score, efficiency ratio) — okrajové zlepšenie (+0,004 AUC), rovnaký
  príbeh o volatilite.
- Extrémy funding rate a kvadranty open interest/cena ako
  kontrariánske signály — štatisticky nevýznamné (p = 0,20 a p = 0,61,
  permutačný test — konvencia použitá v celom tomto repozitári je
  p ≤ 0,05 pre označenie výsledku za významný, takže ani jeden túto
  hranicu neprekonáva).
- Doslovné pravidlo "liquidity sweep / stop hunt" (cena prerazí nedávnu
  swingovú úroveň so súbežným nárastom likvidácií) — na 25 signáloch vo
  vzorke pôsobilo významne (p = 0,008), ale neprešlo kontrolou citlivosti
  na parametre a nepotvrdilo sa na nezávislom datasete z roku 2024
  (spustilo sa tam len 3–4-krát bez ohľadu na parametre).
- Páka 1×–20×, simulovaná s realistickým modelom likvidácie založeným
  na MAE namiesto naivného násobenia PnL — riziko likvidácie bolo pri
  týchto pákach ~0 % pre testované doby držania (1–5 minút), čo
  znamená, že páka jednoducho škáluje už existujúci výsledok; nevytvára
  výhodu a zosilňuje krehkosť marginálnych (PF ≈ 1,0–1,1) konfigurácií.

---

## Jeden reálny trade-off (nie inscenovaný)

Tento projekt vznikol s pomocou AI asistenta pri programovaní.
Neskrývam to — v roku 2026 veľmi nezáleží na tom, či bol kód vytvorený
s pomocou AI; záleží na tom, či osoba za ním dokáže vysvetliť a obhájiť
každé rozhodnutie v ňom. Toto je skutočná výmena z toho procesu,
ponechaná tak, ako bola, pretože lepšie ilustruje podstatu než čokoľvek
napísané dodatočne:

**Otázka: replay L2 order booku trvá ~15 minút pre 90 dní BTC — je to
limit hardvéru, alebo softvéru?**

Väčšinou softvéru. Denný loader v `orderbook_l2.py` parsuje surový JSON
riadok po riadku do obyčajného Python `dict`-based order booku —
nedá sa vektorizovať tak ako zvyšok pipeline (numba-JIT), pretože každý
riadok závisí od nahromadeného stavu tých predchádzajúcich. Skutočná
oprava je jednoduchá a viem presne, aká: 90 dní sú nezávislé súbory bez
zdieľaného stavu medzi sebou, takže ide o učebnicový prípad pre
`multiprocessing.Pool` — paralelizácia naprieč dňami by čas na
2-jadrovom stroji približne prepolovila, takmer zadarmo.

Neurobil som to. Keď výsledok vyšiel čistý — hĺbka order booku nehýbe
s AUC (0,597 → 0,596 → 0,594, šum) — zrýchľovanie kódu, ktorý priniesol
nulový výsledok, prestalo stáť za ten čas. To je reálny trade-off
("funguje a stačí to na záver" vs. "funguje rýchlo"), nie nepovšimnutý
problém — presne tento typ rozhodnutia sa neukáže v diffe, len keď sa
naň niekto opýta priamo.

**Dodatok, 2026-10.** Východisko tejto odpovede bolo nesprávne. Každý
riadok v súboroch order booku je celý order book s 50 úrovňami, nie
delta, takže žiadny riadok nezávisí od predchádzajúcich. Práve čítanie
ako delt pokazilo príznaky hĺbky. Oprava urobila čítanie správnym aj
bezstavovým. Samotný trade-off ako rozhodnutie platí ďalej; je to
pripomienka, že „viem presne, aká je oprava" si zaslúžilo ešte raz
overiť na dátach.

---

## Obmedzenia

Priamo, pretože negatívny výsledok má len takú hodnotu ako podmienky,
za ktorých bol testovaný:

- **Optimistický model plnenia a poplatkov.** Vstup na otvorení
  nasledujúcej sviečky je v praxi trhový (taker) príkaz, no poplatky sú
  modelované maker sadzbou (0,04 % za kruh), bez sklzu. Realistický
  model by každý výsledok tu zhoršil, nie zlepšil.
- **Prevažne jeden symbol.** Batéria ~80 hypotéz bežala na BTCUSDT;
  ETH a SOL len v cross-asset kontrolách. Ostatných 21 zbieraných
  symbolov testovaných nebolo.
- **Viacnásobné porovnania.** Pôvodná štúdia testovala ~80 hypotéz bez
  formálnej korekcie na viacnásobné porovnania. Namiesto nej sa opiera
  o permutačné testy, nezávislý rok a forward test (forward test
  Bonferroniho korekciu používa). Jednotlivé p-hodnoty v reporte treba
  čítať s týmto vedomím.
- **Jedna burza, krátka história.** Iba Bybit, približne pol roka 2026
  plus archív z roku 2024 pre BTC. V iných režimoch a na iných burzách
  to môže byť inak.
- **Minútové sviečky.** Dynamika v rámci minúty (pozícia vo fronte,
  latencia) nie je modelovaná; order book je vzorkovaný ako snímky
  každých ~0,5 s a kolektor ukladá iba časové pečiatky burzy, bez
  lokálneho času prijatia.
- **Dáta nie sú súčasťou repozitára.** Ktokoľvek môže spustiť kód na
  vlastných dátach a syntetická kontrola nižšie overí, že kód funguje,
  no presné čísla tu bez tohto datasetu reprodukovať nejde.

---

## Štruktúra repozitára

```
tickml/               hlavný balík: označovanie, príznaky, loadery, modely, validácia, príznaky L2 order booku
collector/             WS kolektor Bybit, ktorý zozbieral živý dataset (pozri sekciu Dáta)
experiments/           spustiteľné skripty, každý testuje jednu skupinu hypotéz
tests/                 testy správnosti (kontroly look-aheadu, hraničné prípady označovania, matematika validácie)
reports/FINAL_REPORT.md   kompletný číselný prehľad (v angličtine)
reports/runs/          surový výstup opakovaných behov z 2026-10
figures/               PNG obrázky vložené v tomto README
.github/workflows/     CI: testy pri každom pushi
```

## Spustenie

Bez akýchkoľvek dát beží celý pipeline na syntetických tikoch: jeden
svet so zabudovaným zhlukovaním volatility, druhý s konštantnou
volatilitou. Pipeline by mal nájsť to prvé (AUC 0,54–0,62 pri šiestich
seedoch) a v druhom nič (0,50–0,53); rovnaká kontrola beží v CI (`tests/test_synthetic_pipeline.py`).

```bash
uv sync
uv run experiments/00_synthetic_smoke.py
```

S vlastnými dátami:

```bash
uv sync
export TICKML_DATA_DIR=/path/to/your/tick/data
uv run pytest
uv run experiments/01_baseline_and_horizon.py
```

Každý súbor `experiments/*.py` je spustiteľný samostatne a vypisuje
vlastné výsledky; žiadny nevyžaduje, aby predtým bežali ostatné (okrem
lokálnej parquet cache v `.cache/`, ktorá sa zapíše pri prvom načítaní
symbolu, aby sa gigabajty gzipu neparsovali znova pri každom spustení).
`experiments/08_make_figures.py` znovu vygeneruje 3 PNG obrázky
pôvodnej štúdie do `figures/`; `experiments/10_forward_figure.py`
znovu vygeneruje `forward_test.png`.
