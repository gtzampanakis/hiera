from collections import defaultdict
import dataio
import scipy
import scipy.sparse
import scipy.sparse.linalg

INITIAL_RATING = 1500.
RATING_SD = 250.

SURFACES = [
    'Hard',
    'Clay',
    'Grass',
    'Carpet',
]

def get_sind(sstr):
    return SURFACES.index(sstr)

if 'data' not in globals():
    data = list(dataio.iter_tennis_data_xls_data('men'))
# data = [
#     [None, None, '1', '2', 1.5, 6],
#     [None, None, '1', '2', 3, 6],
# ]
    # for d in data:
    #     d[2] = d[2]+d[1]
    #     d[3] = d[3]+d[1]

class Evaluator:

    def __init__(self, data):
        self.data = data

    def pa(self, pname):
        pid = self.pname_to_pid[pname]
        return evor.R[[pid + sind*self.p for sind in xrange(len(SURFACES))]]
    
    def calc_M_D(self):
        pname_to_pid = {}
        pid_to_pname = {}
        pid_to_matches = defaultdict(list)
        match_to_pids = {}
        match_to_pw = {}

        pid = 0
        for di, d in enumerate(self.data):
            p1, p2 = d[2], d[3]
            for name in [p1, p2]:
                if name not in pname_to_pid:
                    pname_to_pid[name] = pid
                    pid_to_pname[pid] = name
                    pid += 1
            match_to_pids[di] = [pname_to_pid[p1], pname_to_pid[p2]]
            pid_to_matches[pname_to_pid[p1]].append(di)
            pid_to_matches[pname_to_pid[p2]].append(di)
            match_to_pw[di] = 1./d[4]

        m = len(self.data)
        p = len(pname_to_pid)

        self.m = m
        self.p = p

# model: M x R = D
# where M is matches (m x p) (sparse)
#       R is ratings (p x 1)
#       D is ratings differences (m x 1)

        csr_data = scipy.zeros(m*2*4)
        csr_row_ind = scipy.zeros(m*2*4)
        csr_col_ind = scipy.zeros(m*2*4)
        for mi, pids in match_to_pids.iteritems():
            rm = 1
            if float(data[mi][6]) > 3:
                rm = 1.3

            mm = [ len(pid_to_matches[pid]) for pid in pids ]
            
            sind = get_sind(data[mi][1])

            for sindc in xrange(len(SURFACES)):
                sm = None
                if sindc == sind:
                    sm = 1.
                else:
                    sm = 0.

                csr_data[mi + sindc*m] = +1 * rm * sm / mm[0]
                csr_row_ind[mi + sindc*m] = mi
                csr_col_ind[mi + sindc*m] = pids[0] + sindc*p

                csr_data[mi + sindc*m + 4*m] = -1 * rm * sm / mm[1]
                csr_row_ind[mi + sindc*m + 4*m] = mi
                csr_col_ind[mi + sindc*m + 4*m] = pids[1] + sindc*p

        M = scipy.sparse.csr_matrix(
            (csr_data, (csr_row_ind, csr_col_ind)), shape=(m,p*4))
        D = scipy.zeros((m,))

        for mi, pw in match_to_pw.iteritems():
            D[mi] = scipy.log(pw / (1-pw)) * RATING_SD

        self.pid_to_pname = pid_to_pname
        self.pname_to_pid = pname_to_pid
        return M, D

    def solve(self, M, D):
        R = scipy.sparse.linalg.lsqr(M, D)[0]
        # order = scipy.argsort(R)
        # for pid in order[-45:]:
        #     print self.pid_to_pname[pid], R[pid]
        return R
    
    def train(self):
        M, D = self.calc_M_D()
        R = self.solve(M, D)
        self.data = self.data
        self.M = M
        self.D = D
        self.R = R

    def predict(self, p1, p2):
        pid1 = self.pname_to_pid[p1]
        pid2 = self.pname_to_pid[p2]
        r1 = self.R[pid1]
        r2 = self.R[pid2]
        diff = r1 - r2
        p = 1./(1. + scipy.exp(-diff/RATING_SD))
        return p

    def loov(self):
        kept_data = self.data
        for i in xrange(len(self.data)):
            self.data = kept_data[:i] + kept_data[i+1:]
            self.train()
            try:
                p_predicted = self.predict(kept_data[i][2], kept_data[i][3])
            except KeyError:
                continue
            else:
                print '%s %s %s %s %.2f - %.2f' % (
                    i, len(self.data),
                    kept_data[i][2], kept_data[i][3],
                    kept_data[i][4],
                    1./p_predicted,
                )
    
evor = Evaluator(data)
evor.train()
prd = evor.M * evor.R - evor.D
print scipy.std(prd)

# print 1./evor.predict('Federer R.', 'Grosjean S.')

