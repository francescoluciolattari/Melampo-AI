# Lettura italiana nel lettore, e perché "T11 level" non si leggeva (15 ottobre 2026)

Misure sugli stessi item del run completo (375 italiani, 484 inglesi), rileggendo con il lettore nuovo le risposte
già date dai modelli. Non è un nuovo run dei modelli: il run vero va rifatto con `role-probe`.

## Perché l'italiano andava male: tre cause, tutte nel lettore

1. **Il lettore leggeva l'italiano con la grammatica inglese.** La lingua arrivava a `read(language=...)` ma non
   cambiava il reticolo: il tipo di un blocco veniva dall'ultima parola (inglese: "heart rate" è un tasso), mentre in
   italiano regge la prima ("frequenza cardiaca", "TC del torace", "asportazione del rene"). In più il linker non
   passava affatto la lingua al lettore.
2. **Le parole italiane erano lette come inglesi.** "su" è in NCIt una sigla chimica, "rx" una proprietà, "rm" e
   "eo" sigle chimiche: "Addome dolente su tutti i quadranti" diventava "nome di una sostanza". NCIt è un
   vocabolario inglese.
3. **Mancava il vocabolario delle procedure italiane**, cioè proprio le parole che reggono la menzione ("TC",
   "radiografia", "ecografia", "intervento").

Provate da sole, né la grammatica né le teste CIPI bastavano (297→288 e 297→293): servono insieme.

## Che cosa è cambiato

- `ChunkLattice.in_language("it")`: il lettore sceglie il reticolo dalla lingua (testa a sinistra, parole
  grammaticali italiane). L'inglese resta com'era. Anche il linker rileva la lingua della frase e la passa.
- Una parola grammaticale (articolo, preposizione) non dà mai il tipo di un blocco; elenco ampliato con
  oltre, sopra, sotto, durante, dopo, prima, verso, presso, senza, mediante, entro…
- In italiano la parola italiana viene prima di NCIt (`BlockMemory.italian`, da `italian_heads.json`: 784 parole
  che reggono almeno 3 titoli CIPI/ICD-9-CM, più la sigla tra parentesi del titolo: "(TAC)", "(OCT)").
- Una struttura coordinata a una struttura ("TC torace **ed** addome") condivide ciò che regge la prima.
- Il predicato del lettore non è mai un avverbio o una parola grammaticale.

## Risultati

| | prima | dopo |
|---|---|---|
| Italiano, lettore giusto | 297/375 (79,2%) | 336/375 (89,6%) |
| Italiano, sedi di procedura viste dal lettore | 0/72 | 54/72 |
| Italiano, Nemotron a una persona | 7 (1,9%) | 5 (1,3%) |
| Italiano, Gemma a una persona | 11 | 9 |
| Italiano, d'accordo senza ruolo comune (Nemotron) | 60 | 23 |
| Italiano, d'accordo ma esito sbagliato | 0 | 0 |
| Inglese, lettore giusto | 458/484 | 458/484 |
| Inglese, Nemotron a una persona | 2 (0,4%) | 1 (0,2%) |
| Inglese, Gemma a una persona | 12 | 11 |

## "T11 level": perché non veniva interpretato

Il lettore aveva due indizi: il blocco "approximately t11 level" (struttura, 1,0) e la **predicazione**, cioè il
vettore della parola vicina nello spazio semantico. Come parola vicina prendeva **"approximately"**, un avverbio
che non dice nulla sulla menzione: il suo vettore puntava a "nome di altra cosa" (1,0). Due indizi uguali e
contrari, 0,501 contro 0,499. Non era "T11 sconosciuto": con "at T11" lo stesso lettore dice struttura senza dubbi.
Ora l'avverbio e le parole grammaticali non sono mai il predicato; "T11 level" → struttura (1,0).

## Che cosa resta (italiano)

Errori del lettore, in ordine di peso:
- **Nominalizzazioni** che in clinica sono reperti ma nei titoli CIPI sono procedure ("dilatazione delle vie
  biliari", "distensione del colon", "identificazione dell'uretere", "occlusione"): il lettore le dà come
  procedura, il modello come struttura. Il contesto (verbo, reperto) deve decidere; oggi è il caso più frequente di
  disaccordo.
- **Forme derivate** non nelle teste: "ecotomografia", "ecografica", "ecocardiogramma", "intervento chirurgico di".
- **"EO"** (esame obiettivo) letto come sigla chimica: manca nel vocabolario italiano delle sigle.
- **Menzioni lontane dalla parola che regge** ("Interruzione della tuba e dei vasi ovarici").

## Da fare
Rilanciare `role-probe` (modelli gemma-4-31b e nemotron-3-ultra, bracci `lattice,reader,llm,llm_method`) per
avere i numeri veri con il lettore nuovo, anche per il metodo `llm_method`.
