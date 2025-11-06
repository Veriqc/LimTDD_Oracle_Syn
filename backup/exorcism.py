from caterpillar_bindings import synthesize_stg_from_tt_json
import json

circ = json.loads(synthesize_stg_from_tt_json(6, 0, 127))
for gate in circ:
    print(gate)