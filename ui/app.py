from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
from datetime import datetime

app = Flask(__name__)

RESULTS_PATH    = "../data/processed/coding_results.csv"
DECISIONS_PATH  = "../data/processed/coder_decisions.csv"

def load_results():
    df = pd.read_csv(RESULTS_PATH)
    df = df[df['parse_success'] == True].reset_index(drop=True)
    return df

def load_decisions():
    if os.path.exists(DECISIONS_PATH):
        return pd.read_csv(DECISIONS_PATH)
    return pd.DataFrame(columns=[
        'note_id', 'code', 'description',
        'reason', 'decision', 'decided_at'
    ])

@app.route('/')
def index():
    df       = load_results()
    decisions = load_decisions()

    notes = []
    for i, row in df.iterrows():
        note_id    = row['note_id']
        codes      = json.loads(row['selected_codes'])
        decided    = decisions[decisions['note_id'] == note_id]
        reviewed   = len(decided) > 0
        accepted   = len(decided[decided['decision'] == 'accept']) if reviewed else 0
        rejected   = len(decided[decided['decision'] == 'reject']) if reviewed else 0

        notes.append({
            'index':    i,
            'note_id':  note_id,
            'preview':  row['note_preview'][:100],
            'total':    len(codes),
            'reviewed': reviewed,
            'accepted': accepted,
            'rejected': rejected
        })

    return render_template('index.html', notes=notes, total=len(notes))

@app.route('/review/<int:note_index>')
def review(note_index):
    df        = load_results()
    decisions = load_decisions()

    if note_index >= len(df):
        return "Note not found", 404

    row     = df.iloc[note_index]
    note_id = row['note_id']
    codes   = json.loads(row['selected_codes'])

    # Load the full note from the cleaned PMC dataset
    pmc_df    = pd.read_csv("../data/pmc_processed/pmc_cleaned.csv")
    pmc_row   = pmc_df[pmc_df['idx'] == note_id]

    if len(pmc_row) > 0:
        # Show exactly what was sent to the model — first 8000 chars
        full_note = pmc_row.iloc[0]['full_note'][:1500]
    else:
        full_note = row['note_preview']

    decided = decisions[decisions['note_id'] == note_id]
    decided_codes = {}
    for _, d in decided.iterrows():
        decided_codes[d['code']] = d['decision']

    total  = len(df)
    prev_i = note_index - 1 if note_index > 0 else None
    next_i = note_index + 1 if note_index < total - 1 else None

    return render_template('review.html',
        note_index  = note_index,
        note_id     = note_id,
        note_text   = full_note,
        codes       = codes,
        decided     = decided_codes,
        prev_i      = prev_i,
        next_i      = next_i,
        total       = total
    )

@app.route('/decide', methods=['POST'])
def decide():
    data     = request.json
    note_id  = data['note_id']
    code     = data['code']
    decision = data['decision']

    decisions = load_decisions()

    mask = (decisions['note_id'] == note_id) & (decisions['code'] == code)
    if mask.any():
        decisions.loc[mask, 'decision']   = decision
        decisions.loc[mask, 'decided_at'] = datetime.now().isoformat()
    else:
        new_row = pd.DataFrame([{
            'note_id':    note_id,
            'code':       code,
            'description': data.get('description', ''),
            'reason':     data.get('reason', ''),
            'decision':   decision,
            'decided_at': datetime.now().isoformat()
        }])
        decisions = pd.concat([decisions, new_row], ignore_index=True)

    decisions.to_csv(DECISIONS_PATH, index=False)
    return jsonify({'status': 'ok', 'decision': decision})

@app.route('/stats')
def stats():
    decisions = load_decisions()
    if len(decisions) == 0:
        return jsonify({'total': 0, 'accepted': 0, 'rejected': 0})

    return jsonify({
        'total':    len(decisions),
        'accepted': len(decisions[decisions['decision'] == 'accept']),
        'rejected': len(decisions[decisions['decision'] == 'reject'])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)
