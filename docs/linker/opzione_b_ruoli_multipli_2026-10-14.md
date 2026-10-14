# Opzione b: ruoli multipli, accordo sull'esito, link anche per la misura (14 ottobre 2026)

Decisione di Frank: il contesto sceglie, non una regola per caso. Più ruoli possono essere veri insieme. Il lettore e il modello devono concordare su ciò che cambia l'esito.

## Cosa cambia

**1. La domanda al modello** (`llm_reading`). La risposta ha `"roles": [...]`, cioè tutti i ruoli che la frase sostiene, il più importante per primo, al posto di un solo `"role"`. La regola 5 del prompt lo spiega con tre esempi:

- una misura di un organo che è anche un reperto in esso;
- il luogo di una raccolta che è anche la sua origine;
- una sede esaminata che è anche la sede di un reperto.

La definizione di `procedure_site` dice ora che la procedura può essere nominata dalla parola che regge la menzione, anche fuori dal sintagma ("liberation of the caecum and of the appendix"). Le risposte con il vecchio formato (`"role"`) restano leggibili. Il prompt è cambiato, quindi le risposte in cache non valgono: il prossimo run fa chiamate nuove.

**2. La regola di accordo** (`role_probe.site_agreement`):

- lettore e modello devono concordare su **sede del corpo sì o no**;
- i ruoli fini si registrano come insieme: quelli che il lettore tiene possibili (quota ≥ 0,2) e quelli che il modello indica;
- una persona interviene solo quando i due si contraddicono sull'esito, o quando uno dei due non sa.

Il report ha una tabella nuova, per lingua e per modello:

- item mandati a una persona;
- item concordi ma con esito sbagliato;
- item concordi con almeno un ruolo in comune;
- item concordi senza ruoli in comune.

**3. Il linker** (`AnatomyLinker.link_measured_structures`, attivo per default). Una struttura di cui la frase nomina una misura o una funzione ("heart rate", "funzione del fegato", "heart development") ora **è collegata**. Il ruolo `inherent_location` va insieme al link (`LinkResult.roles`); prima il linker si asteneva e scriveva solo `about`.

- Il collegamento si fa solo se nessun altro motivo blocca il link (lingua del nome, frame non clinico…).
- `LinkResult.roles` porta tutti i ruoli: per esempio "inherent_location" più "procedure_site".
- Chi usa il link legge i ruoli: "normal heart size" non è un reperto nel cuore.
- `link_measured_structures=False` ripristina l'astensione.

Nella valutazione sul gold set (`gold_set.evaluate`), un link con ruolo misura su una riga NOT_ANATOMY/inherent_location ha un esito a parte, `measured_structure`. Non è un errore critico; la classe non si verifica, perché la riga non la indica.

## Cosa dicono le risposte già raccolte (run del 13 ottobre, prima della nuova domanda)

Item mandati a una persona con la regola sull'esito:

| | inglese, 484 item | italiano, 375 item |
|---|---|---|
| Gemma 4 31B | 9 (1,9%) | 17 (4,5%) |
| Nemotron 3 Ultra | 9 (1,9%), **3 (0,6%)** con le correzioni del controllo | 7 (1,9%) |
| gpt-oss-120b FP8 | 43 (8,9%) | 13 (3,5%) |

- **Concordi ma con esito sbagliato: 0** in tutte le righe.
- Tra gli item concordi, in inglese il 7–10% non ha un ruolo in comune tra lettore e modello (per esempio struttura contro sede di procedura). Lo si guarda con la nuova domanda, in cui il modello può dare più ruoli.

## Da misurare

Il run `role-probe` con la nuova domanda: Gemma 4 31B e Nemotron 3 Ultra, `governor` = `false`, senza `resume_run`, perché il prompt è cambiato.

## Il run del 14 ottobre sera (nuova domanda, Gemma 4 31B e Nemotron 3 Ultra)

Item mandati a una persona (lettore e modello si contraddicono su "sede del corpo sì o no", oppure uno dei due non sa):

| | inglese, 484 item | italiano, 375 item |
|---|---|---|
| **Nemotron 3 Ultra** | **2 (0,4%)** | 9 (2,4%); **7 (1,9%)** con la correzione del controllo qui sotto |
| Gemma 4 31B | 12 (2,5%) | 20 (5,3%); 11 (2,9%) con la correzione |

- **Concordi ma con esito sbagliato: 0** in tutte le righe.
- Il modello dà più di un ruolo in circa un terzo delle risposte: Nemotron 302 volte su 963, Gemma 272.

**L'inglese con Nemotron è sotto l'1%.** I 2 casi che restano sono una frase mozzata ("Granulomatous disease in the bilateral") e "T11 level", dove il lettore non sa.

**Gemma va peggio con la nuova domanda.** Da sola dà il primo ruolo giusto l'85,8% delle volte (era 91,1%), con 14 errori sede/non-sede. Per esempio legge "pleural" in "No pleural effusion" come parte del nome di un'altra cosa (`inside_a_name`), 3 volte. Per questo compito Nemotron è chiaramente migliore.

**Correzione del controllo.** Nei referti italiani il gold include l'articolo o la preposizione ("nelle membrane basali", "del tronco celiaco"). Il modello li lascia fuori dal sintagma ("membrane basali glomerulari"), e il controllo scartava la risposta. Ora articoli e preposizioni ai bordi della menzione possono restare fuori; anche "mesenterico- portale" e "mesenterico-portale" valgono come la stessa scrittura. Sulle risposte già pagate sono 11 risposte recuperate, e i numeri "con la correzione" vengono da lì.

**Cosa resta in italiano (Nemotron, 7):**

- 3 sono errori del lettore, che legge "parte di un nome" dove c'è una sede: "Addome dolente", "al torace", "dell'ovaio destro" in "Asportazione di cisti dermoide dell'ovaio destro". Il lettore non ha un vocabolario italiano: è il lavoro su CIPI, ICD-9-CM e MeSH italiano.
- 1 è "cuffia" (dei rotatori), su cui il lettore non sa.
- 1 è "MO": il modello ha risposto con l'espansione "midollo osseo".
- 2 sono risposte del modello che non si possono usare.

**Concordi senza un ruolo in comune:** 12 in inglese e 60 in italiano (Nemotron). Sono quasi tutte sedi di procedura: il modello dice {sede di procedura, struttura}, il lettore solo "struttura" con quota piena. L'esito non cambia (sede), ma il ruolo "procedura" lo vede solo il modello. Anche qui serve il vocabolario italiano del lettore.

### Perché Gemma sembrava peggiorata, e cosa vale anche per Nemotron (analisi)

**La misura del primo ruolo era fuorviante.** Con più ruoli, "primo ruolo giusto" non misura più l'accuratezza.

- Gemma passa dal 91,1% all'85,8% sul primo ruolo, ma il ruolo giusto è **tra quelli che indica nel 95,9%** dei casi.
- Nemotron: 94,7% sul primo ruolo, **98,4%** tra quelli indicati.

Dei 63 item che Gemma "ha perso", 52 danno il ruolo giusto per secondo. Per esempio {sede di procedura, struttura} per "the testis" in "we removed the sac after separating the vas and vessels and the testis". Il report ora riporta entrambe le misure.

**L'errore vero (14 item sede/non-sede, tutti di Gemma) viene dal nostro prompt.** In 11 casi su 14 Gemma dice `inside_a_name` per un reperto o una malattia che porta il nome della sede:

- "pleural effusion" (3), "pulmonary edema", "versamento pleurico";
- "osteosarcoma del femore destro", "pucker maculare", "membrana neovascolare coroideale";
- "bronchovascular crowding", "head posture".

La regola 3 diceva che una struttura scritta dentro il nome più lungo di un'altra cosa fa parte di quel nome. La definizione di `inside_a_name` finiva con "…o altra cosa". Gemma ha letto "pleural effusion" come il nome di un'altra cosa.

**Corretto nelle definizioni, senza regole caso per caso:**

- `inside_a_name` vale solo per il nome di una cosa che non è nel corpo: gene, proteina, sostanza, organismo, organizzazione, prodotto;
- un reperto, una malattia, una procedura o una misura che portano il nome della sede si riferiscono ancora alla struttura.

Nemotron non ha sbagliato su questo in questo run, ma la definizione ambigua valeva anche per lui.

**I 12 casi inglesi "concordi senza un ruolo in comune" (Nemotron) non sono errori di esito.** Lettore e modello concordano che è una sede, quindi non serve una persona; differisce solo il ruolo fine.

- In 10 casi su 12 ha ragione il modello:
  - "CT brain", "bone marrow exam", "brain … MRI", "valve replacement": sedi di procedura che il lettore non vede;
  - "the cervix looked", "lower lobes", "costophrenic", "both hips", "knees pain", "cardiomediastinal silhouette": tipi NCIt sbagliati di parole vicine.
- In 2 casi ha ragione il lettore: "hair" in "color of the hair" e "aorta" in "tortuosity of the aorta".
