import copy
import json

from tools.robustness import lp_survival as survival


def _config():
    return {
        'outside_option': 1.3319e-9,
        'subsidy_rate': 0.0,
        'loss_rebate_fraction': 0.0,
        'exit_patience': 290,
        'entry_patience': 290,
        'kappa': 0.35,
        'response_scale': 1e-6,
        'max_adj': 0.002,
        'ewma_alpha': 0.02,
        'entry_margin': 0.25,
        'n_iter': 1000,
    }


def _pool_row():
    row = {key: False for key in survival._POOL_BOOL_FIELDS}
    row.update({key: 0 for key in survival._POOL_INT_FIELDS})
    row.update({key: None for key in survival._POOL_NULLABLE_INT_FIELDS})
    row.update({key: 0.0 for key in survival._POOL_FLOAT_FIELDS})
    row['pool'] = 'hfmm'
    return row


def _record(seed=42, signature='survival-test-signature'):
    return {
        'raw_schema_version': survival._RAW_SCHEMA_VERSION,
        'seed': seed,
        'shock': 350,
        'pools': [_pool_row()],
        'model_signature': survival.model_signature(survival.ROOT),
        'signature': signature,
        'configuration': _config(),
    }


def _load(tmp_path, rows, signature='survival-test-signature'):
    path = tmp_path / 'survival.jsonl'
    text = ''
    for row in rows:
        text += row if isinstance(row, str) else json.dumps(row)
        text += '\n'
    path.write_text(text, encoding='utf-8')
    return survival.load_raw(str(path), signature, _config())


def test_survival_raw_schema_accepts_only_one_canonical_hfmm_row(tmp_path):
    valid = _record()
    have, invalid = _load(tmp_path, [valid])

    assert set(have) == {42}
    assert invalid == 0

    duplicate_pool = copy.deepcopy(valid)
    duplicate_pool['seed'] = 1
    duplicate_pool['pools'].append(copy.deepcopy(duplicate_pool['pools'][0]))
    missing_pool = copy.deepcopy(valid)
    missing_pool['seed'] = 2
    missing_pool['pools'] = []
    wrong_pool = copy.deepcopy(valid)
    wrong_pool['seed'] = 3
    wrong_pool['pools'][0]['pool'] = 'cpmm'

    have, invalid = _load(tmp_path, [duplicate_pool, missing_pool, wrong_pool])

    assert have == {}
    assert invalid == 3


def test_survival_raw_schema_rejects_and_counts_every_malformed_identity(tmp_path):
    valid = _record(seed=100)
    missing_schema = copy.deepcopy(valid)
    missing_schema['seed'] = 1
    missing_schema.pop('raw_schema_version')
    wrong_schema = copy.deepcopy(valid)
    wrong_schema['seed'] = 2
    wrong_schema['raw_schema_version'] += 1
    wrong_signature = copy.deepcopy(valid)
    wrong_signature['seed'] = 3
    wrong_signature['signature'] = 'stale'
    wrong_model = copy.deepcopy(valid)
    wrong_model['seed'] = 4
    wrong_model['model_signature'] = 'stale'
    wrong_config = copy.deepcopy(valid)
    wrong_config['seed'] = 5
    wrong_config['configuration']['entry_margin'] = 0.5
    wrong_config_type = copy.deepcopy(valid)
    wrong_config_type['seed'] = 6
    wrong_config_type['configuration']['n_iter'] = 1000.0
    wrong_seed_type = copy.deepcopy(valid)
    wrong_seed_type['seed'] = 7.0
    wrong_shock_type = copy.deepcopy(valid)
    wrong_shock_type['seed'] = 8
    wrong_shock_type['shock'] = True
    missing_identity = copy.deepcopy(valid)
    missing_identity['seed'] = 9
    missing_identity['pools'][0].pop('gross_event_identity_holds')
    wrong_identity_type = copy.deepcopy(valid)
    wrong_identity_type['seed'] = 10
    wrong_identity_type['pools'][0]['claim_supply_identity_end_holds'] = 1
    wrong_float_type = copy.deepcopy(valid)
    wrong_float_type['seed'] = 11
    wrong_float_type['pools'][0]['active_end'] = 0
    nonfinite = copy.deepcopy(valid)
    nonfinite['seed'] = 12
    nonfinite['pools'][0]['rho_over_option'] = float('nan')

    have, invalid = _load(tmp_path, [
        valid,
        '{not-json}',
        missing_schema,
        wrong_schema,
        wrong_signature,
        wrong_model,
        wrong_config,
        wrong_config_type,
        wrong_seed_type,
        wrong_shock_type,
        missing_identity,
        wrong_identity_type,
        wrong_float_type,
        nonfinite,
    ])

    assert set(have) == {100}
    assert invalid == 13


def test_survival_raw_duplicate_seed_invalidates_both_rows(tmp_path):
    first = _record(seed=42)
    duplicate = copy.deepcopy(first)
    unique = _record(seed=43)

    have, invalid = _load(tmp_path, [first, duplicate, unique])

    assert set(have) == {43}
    assert invalid == 2


def test_survival_cache_path_is_measurement_signature_scoped():
    assert survival._raw_path(_config(), 'a' * 16) != survival._raw_path(
        _config(), 'b' * 16
    )


def test_survival_json_cannot_be_complete_with_rejected_raw_rows():
    rec = _record(signature=survival.survival_signature())
    payload = survival._survival_json_payload(
        [rec], _config(), survival.survival_signature(),
        survival.survival_report_signature(), 42, [42], 1,
    )

    assert payload['available_seeds'] == 1
    assert payload['invalid_or_stale_records_ignored'] == 1
    assert payload['raw_provenance_clean'] is False
    assert payload['complete'] is False


def test_missing_survival_identities_never_pass_by_default(capsys):
    summary = survival.summarize([
        {
            'seed': 42,
            'pools': [{
                'pool': 'hfmm',
                'open_at_shock': True,
                'open_through_crisis': True,
                'first_close': None,
            }],
        },
        {
            'seed': 43,
            'pools': [{
                'pool': 'hfmm',
                'open_at_shock': True,
                'open_through_crisis': True,
                'first_close': None,
                'gross_event_identity_holds': 1,
                'claim_supply_identity_end_holds': 'true',
            }],
        },
    ], {'n_iter': 1000})['pools']['hfmm']

    assert summary['gross_event_identity_pass_count'] == 0
    assert summary['gross_event_identity_pass_rate'] == 0.0
    assert summary['claim_supply_identity_end_pass_count'] == 0
    assert summary['claim_supply_identity_end_pass_rate'] == 0.0

    row = _pool_row()
    row.pop('gross_event_identity_holds')
    row.pop('claim_supply_identity_end_holds')
    text = survival.report(
        [{'seed': 42, 'pools': [row]}], _config()
    )
    capsys.readouterr()
    assert '1 event / 1 claim-supply' in text
