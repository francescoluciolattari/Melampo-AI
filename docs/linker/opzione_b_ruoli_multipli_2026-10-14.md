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
