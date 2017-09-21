import glob
import os
import xlrd

import hieramath
import hieraconfig as hc

HEADER = [
    'date',
    'surface',
    'winner',
    'loser',
    'oddsw',
    'oddsl',
    'raceto',
]

def iter_tennis_data_xls_data(gender):
    dirname = os.path.join(hc.TENNIS_DATA_XLS_DIR, gender)
    for path in glob.glob(os.path.join(dirname, '2[0-9][0-9][0-9]*.xls*')):
        wb = xlrd.open_workbook(path)
        try:
            sheet = wb.sheet_by_name(os.path.basename(path)[:4])
        except xlrd.biffh.XLRDError as e:
            sheet = wb.sheet_by_index(0)
        header = [sheet.cell_value(0,c) for c in xrange(sheet.ncols)]
        odds_indices = []
        for i in xrange(len(header)):
            if header[i].endswith('W'):
                if header[i+1].endswith('L'):
                    if header[i][:-1] == header[i+1][:-1]:
                        bookie_code = header[i][:-1]
                        if (
                                 bookie_code.lower() != 'avg' 
                             and bookie_code.lower() != 'min'
                             and bookie_code.lower() != 'max'
                        ):
                            odds_indices.append(i)
        for rowi in xrange(1, sheet.nrows):
            def val(h):
                return sheet.cell_value(rowi, header.index(h))
            if val('Comment') != 'Completed':
                continue
            row = []
            date_val = val('Date')
            date_val = xlrd.xldate.xldate_as_datetime(date_val, 0).date()
            row.append(date_val)
            row.append(val('Surface').strip())
            row.append(val('Winner').strip())
            row.append(val('Loser').strip())
            sw = sl = 0
            for oi in odds_indices:
                oddsw = sheet.cell_value(rowi, oi)
                oddsl = sheet.cell_value(rowi, oi+1)
                if (
                    oddsw and oddsl 
                    and isinstance(oddsw, float) and isinstance(oddsl, float)
                ):
                    pw, pl = 1./oddsw, 1./oddsl
                    pw, pl = pw/(pw+pl), pl/(pw+pl)
                    oddsw, oddsl = 1./pw, 1./pl
                    oddsw, oddsl = hieramath.normalize_odds(oddsw, oddsl)
                    sw += oddsw
                    sl += oddsl
            if sw > 0 and sl > 0:
                oddsw = sw/len(odds_indices)
                oddsl = sl/len(odds_indices)
                oddsw, oddsl = hieramath.normalize_odds(oddsw, oddsl)
                row.append(oddsw)
                row.append(oddsl)
                row.append(val('Best of'))
                yield row

