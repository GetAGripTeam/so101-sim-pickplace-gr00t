# Hackathon-Setup

Die Simulation und das Training laufen auf der Nebius-VM, nicht auf dem Mac.

- Projekt: `group38-project` (`<project-id>`)
- VM: `so101-sim` (`<vm-id>`)
- GPU: NVIDIA RTX PRO 6000 Blackwell, 96 GB
- Boot-Disk: `<disk-id>`, 300 GB
- Öffentliche IP: `<VM-IP>` (kann sich nach Stop/Start ändern)
- Angezeigte Schätzung im Erstellungsassistenten: ca. 1,83 USD/Stunde

## Verbinden und Fortschritt prüfen

```bash
ssh ubuntu@<VM-IP>
tail -f ~/work/so101_sim_pickplace/logs/collect.log
```

Validiert: GPU-Berechnung, GR00T-Basismodellserver, Front- und Handgelenkkamera,
drei erfolgreiche Demonstrationen und GR00T-Datensatzprüfung (`DATASET OK`).
Das Modell wurde noch nicht auf SO-101 trainiert.

Das Projekt liegt in `~/work/so101_sim_pickplace`, die beiden Python-Umgebungen
in `~/work/env_isaaclab` und `~/work/Isaac-GR00T/.venv`.
Maschinenpfade stehen in `env.local.sh` (nicht in Git).

## Modelle freischalten und herunterladen

Mit dem eigenen Hugging-Face-Account die Bedingungen auf
https://huggingface.co/nvidia/Cosmos-Reason2-2B akzeptieren.
Den Token nur im interaktiven Login auf der VM eingeben, nicht im Chat oder in Git.

```bash
cd ~/work/Isaac-GR00T
~/.local/bin/uv run hf auth login
~/.local/bin/uv run hf download nvidia/GR00T-N1.7-3B --local-dir checkpoints/GR00T-N1.7-3B
```

## Weiterarbeiten nach dem Setup

```bash
cd ~/work/so101_sim_pickplace
NUM_EPISODES=100 ./collect.sh --resume
./prepare_dataset.sh
GLOBAL_BATCH_SIZE=8 MAX_STEPS=4000 ./finetune.sh
```

Der Setup-Test hat bereits drei erfolgreiche Episoden (917 Frames) erzeugt.
`NUM_EPISODES=100` mit `--resume` sammelt 100 weitere Episoden.

Danach in zwei SSH-Terminals:

```bash
cd ~/work/so101_sim_pickplace
./start_server.sh
```

```bash
cd ~/work/so101_sim_pickplace
./eval.sh --num_episodes 20 --record_dir logs/eval
```

## Kosten beenden

Nach dem Hackathon die VM in der Nebius-Konsole stoppen. Die SSD bleibt dabei
bestehen und kann weiter Kosten verursachen. VM und Disk erst löschen, wenn
Datensätze und Checkpoints gesichert sind.

## Automatischer Trainingslauf

Nach Erhalt der vollständigen Challenge-Regeln wurde der automatische Trainingsanschluss gestoppt. Die bereits gestartete Sammlung läuft bis insgesamt 100 erfolgreiche Sim-Episoden weiter. Vor dem Haupttraining fehlen Cosmos Transfer, Cosmos-3-Kuratierung und echte Teleop-Episoden. `scripts/train_pipeline.sh` wartet nun standardmäßig nach Sammlung und Datensatzprüfung auf diese Daten. `TRAIN_AFTER_COLLECTION=1` startet ausdrücklich eine Sim-Baseline.
Ein separater Test mit zwei Trainingsschritten wurde erfolgreich abgeschlossen;
sein Checkpoint liegt unter `logs/train-preflight/checkpoint-2` und ist nur ein
technischer Funktionstest.

```bash
ssh ubuntu@<VM-IP>
cd ~/work/so101_sim_pickplace
cat logs/train-pipeline.status
tail -f logs/train-pipeline.log
```

Der produktive Checkpoint wird unter `finetune/so101_sim_pickplace/` gespeichert.
Der Lauf bleibt nach dem Schließen des SSH-Terminals aktiv. Er ist beendet, wenn
in der Statusdatei `complete` steht; bei `failed` das Pipeline-Log prüfen.
Ein erneuter Aufruf von `scripts/train_pipeline.sh` verhindert parallele Läufe
über eine Dateisperre, startet aber nach Abschluss ein neues Training; daher
nicht während dieses Laufs noch manuell `finetune.sh` aufrufen.

## Challenge-Anforderungen

- Maximal 300 Trainingsepisoden insgesamt, maximal 50 davon echte Teleop-Episoden.
- Hauptmodell: Sim-abgeleitete Daten plus Cosmos-Augmentierung und echte Demos.
- Ablation: dieselbe Policy und gleiche Trainingsparameter ohne Cosmos.
- Cosmos Transfer 2.5: Edge-Konditionierung, Kamerazeiten und Aktionen erhalten; das Aussehen der realen Station treffen.
- Cosmos 3 Reasoner: eigene Episodenvideos bewerten, echte JSON-Ausgaben speichern und mindestens eine daraus abgeleitete Auswahlentscheidung dokumentieren. Der GR00T-Cosmos-Reason2-Backbone erfüllt diesen Punkt nicht.
- Quellenmanifest: `python3 scripts/dataset_manifest.py --source sim=datasets/local/so101_sim_pickplace --output logs/dataset-manifest.json`; weitere Quellen mit `--source cosmos=...` und `--source real=...` hinzufügen. Das Skript inventarisiert Dateien und prüft Episodenlimits; es ersetzt keine Cosmos-Provenienz.
- Abgabe: offizieller Nebius-Object-Storage-Pfad und Cutoff noch bei der Organisation zu klären. Im Teamprojekt war bei Prüfung kein Bucket vorhanden.
- Neun Würfelpositionen an der echten Station berücksichtigen; drei Judge-Trials, maximal zwei Minuten pro Trial.

Empfohlener Vergleich: gleiche Sim-Ursprungsepisoden und gleiche Real-Demos in beiden Läufen; im Hauptlauf die Sim-Videos durch zeitlich passende Cosmos-Versionen ersetzen. So bleibt die Episodenanzahl gleich. Aufbereitungsvarianten auf dieselbe Ursprungs-ID abbilden und nach Ursprungs-Episode splitten.

## NVIDIA-Referenzen für die nächsten Schritte

- Kurs: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html
- Echte Aufnahmen und Co-Training: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/13-strategy2-cotraining.html
- Cosmos: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/14-strategy3-cosmos.html
- Kalibrierung: https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/07-calibrating-so101.html
- Skills-Katalog: https://github.com/NVIDIA/skills

Der Kurs behandelt Röhrchen und ein Rack. Seine vortrainierten Modelle und
Beispieldatensätze sind daher kein fertiges Modell für unsere Würfelaufgabe.
Die realen Beispiele nutzen einen Linux-NVIDIA-Docker-Host; diese Docker-Befehle
nicht unverändert auf dem Apple-Silicon-Mac ausführen. Für uns bleibt der
GPU-Modellserver auf Nebius, während die Hardwareanbindung lokal erfolgt.

Die Beispiele verwenden `front` und `wrist` mit 640×480 bei 30 FPS – passend zu
unserem Sim-Datensatz. Beim Aufnehmen müssen Gelenkreihenfolge, Einheiten,
Kameranamen und Zeitzuordnung ebenfalls übereinstimmen. Das Ausgabeformat der
gewählten LeRobot-Version vor dem Mischen gegen unseren v2.1-Loader prüfen.

Das Cosmos-Kapitel zeigt Edge-Konditionierung mit Gewicht 1.0 und optionale
weitere Kontrollen. Für die Challenge nur Aussehen variieren: keine Änderung
von Würfelposition, Bewegungsverlauf oder Kamerageometrie, solange die Aktionen
unverändert weiterverwendet werden. Generierte Videos vor dem Training auf
Struktur und Frame-Zuordnung prüfen. Die vorab erzeugten Kursbeispiele ersetzen
nicht unseren eigenen Cosmos-Transfer- und Reasoner-Nachweis.

Der Skill `physical-ai-video-data-augmentation` beschreibt OSMO-Workflows mit
OSMO-Zugang, Storage und GPU-Pool. Diese Infrastruktur ist in unserem bisherigen
VM-Setup nicht eingerichtet. Der Katalog allein ist kein Cosmos-API-Zugang.
