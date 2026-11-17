"""Plain-language warnings for the web app: python -m pytest tabvid/tests/test_web_plain.py -q"""
from tabvid.web.plain import explain, explain_one


def test_numbers_and_measures_in_the_sentence_come_from_the_message():
    w = explain_one('numbers', '20 measure(s) missing between m40 (number 6) and m44 (number 30): a barline '
                               'missed (a measure holding two), or frames/pages dropped.')
    assert w['level'] == 'check' and 'between measures 40 and 44' in w['text'] and 'from 6 to 30' in w['text']
    assert w['detail'].startswith('20 measure(s) missing')
    w = explain_one('measures', 'wide measures: 12 (364px), 16 (364px), over 1.6x the typical 188px and wide for '
                                'their row too, with no other sign of a missing barline. Probably long measures; '
                                'a barline may have been missed. Kept as they are.')
    assert w['level'] == 'note' and w['text'].startswith('Measures 12 and 16 are unusually long')
    w = explain_one('measures', 'measure 41 (row 15) is 1187px wide vs a typical 522px and a full-height stroke '
                                '534px, 597px into it failed the barline rule: a barline inside it was probably '
                                'missed. Kept as one measure.')
    assert 'Measure 41 is about 2.3 times as wide' in w['text']


def test_frame_times_are_shown_as_video_times_after_the_range_start():
    message = ("frames ['#7-#10 (3.5-5.0s, middle)', '#14-#14 (7.0-7.0s, middle)'] (5 frames) show long horizontal "
               'lines but no TAB staff was found in them: music shown only there is missing from the sheet '
               '(staff lines broken up, or the staff cut by the crop: try --recrop).')
    assert 'at 0:04-0:05 and 0:07 the video' in explain_one('rows', message)['text']
    assert 'at 1:34-1:35 and 1:37 the video' in explain_one('rows', message, offset=90)['text']


def test_a_failed_stage_is_a_problem_and_problems_come_first():
    out = explain([{'stage': 'sheet', 'message': 'line 4 is stretched 1.70x (over 1.4); no better line break was '
                                                 'found; 0px of it is blank staff before barlines (the music has '
                                                 'too few flat gaps).'},
                   {'stage': 'rows', 'message': 'stage 3 failed (ValueError: boom); wrote a one-page PDF with the error.'}])
    assert [w['level'] for w in out] == ['problem', 'note']
    assert 'step 3 of 5' in out[0]['text'] and out[1]['text'] == 'Line 4 is spread out more than usual (the music is dense there).'


def test_an_unknown_warning_is_shown_as_it_is():
    w = explain_one('rows', 'something new happened at row 3.')
    assert w['level'] == 'check' and w['text'] == 'Something new happened at row 3.' and w['unmatched']


def test_the_same_words_in_another_stage_do_not_match():
    w = explain_one('measures', 'measure 9 (row 2) is 40px wide vs a typical 500px and holds no notes: not a measure '
                                '(a time signature between two barlines, or the cut-off end of a page).')
    assert 'times as wide' in w['text']             # the measures rule, not the structure one
    w = explain_one('structure', 'measure 9 (row 2) is 40px wide vs a typical 500px and holds no notes: not a measure '
                                 '(a time signature between two barlines, or the cut-off end of a page).')
    assert w['text'] == 'Measure 9 holds no notes (a time signature, or the cut-off end of a page).'


def test_a_dropped_seam_repeat_is_a_note():
    w = explain_one('measures', 'strip 2: m66-m69 = m62-m65 (numbers 62-65): the scroll join found no overlap there, '
                                'but the new strip starts by repeating the old one measure for measure, printed '
                                'numbers included; the repeats were dropped.')
    assert w['level'] == 'note' and w['text'] == 'The video showed measures 62-65 twice where it scrolled; the copy was left out.'
