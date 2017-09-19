from collections import defaultdict
import dataio
import scipy
import scipy.sparse
import scipy.sparse.linalg

INITIAL_RATING = 1500.
RATING_SD = 250.

data = list(dataio.iter_tennis_data_xls_data('men'))

print len(data)

pname_to_pid = {}
pid_to_pname = {}
pid_to_matches = defaultdict(list)
match_to_pids = {}
match_to_pw = {}

pid = 0
for di, d in enumerate(data):
    p1, p2 = d[1], d[2]
    for name in [p1, p2]:
        if name not in pname_to_pid:
            pname_to_pid[name] = pid
            pid_to_pname[pid] = name
            pid += 1
        p_to_matches[pname_to_pid[name]].append(di)
    match_to_pids[di] = [pname_to_pid[p1], pname_to_pid[p2]]
    match_to_pw[di] = 1./d[4]

m = len(data)
p = len(pname_to_pid)

# model: M x R = D
# where M is matches (m x p) (sparse)
#       R is ratings (p x 1)
#       D is ratings differences (m x 1)

csr_data = scipy.zeros(m*2)
csr_row_ind = scipy.zeros(m*2)
csr_col_ind = scipy.zeros(m*2)
for mi, pids in match_to_pids.iteritems():
    csr_data[mi] = +1
    csr_row_ind[mi] = mi
    csr_col_ind[mi] = pids[0]

    csr_data[mi+m] = -1
    csr_row_ind[mi+m] = mi
    csr_col_ind[mi+m] = pids[1]

M = scipy.sparse.csr_matrix((csr_data, (csr_row_ind, csr_col_ind)))
D = scipy.zeros((m,))

for mi, pw in match_to_pw.iteritems():
    D[mi] = scipy.log(pw / (1-pw)) * RATING_SD

R = scipy.sparse.linalg.lsqr(M, D)[0]
