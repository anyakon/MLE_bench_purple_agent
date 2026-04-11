import sys, io, os, tarfile, tempfile
sys.path.insert(0, 'src')
import pandas as pd, numpy as np
from purple_agent import PurpleAgent

np.random.seed(42)
train = pd.DataFrame({
    'PassengerId': [f'{i:04d}' for i in range(50)],
    'Age': np.random.uniform(0, 80, 50),
    'Transported': np.random.choice([True, False], 50),
})
test = pd.DataFrame({
    'PassengerId': [f'{i:04d}' for i in range(50, 55)],
    'Age': np.random.uniform(0, 80, 5),
})

td = tempfile.mkdtemp()
dd = os.path.join(td, 'home', 'data')
os.makedirs(dd)
train.to_csv(os.path.join(dd, 'train.csv'), index=False)
test.to_csv(os.path.join(dd, 'test.csv'), index=False)
tp = os.path.join(td, 'c.tar.gz')
with tarfile.open(tp, 'w:gz') as t:
    t.add(os.path.join(td, 'home'), arcname='home')
with open(tp, 'rb') as f:
    tb = f.read()

a = PurpleAgent()
sb = a.solve_competition(tb)
a.cleanup()

df = pd.read_csv(io.BytesIO(sb))

result_file = os.path.join(td, '_output.txt')
with open(result_file, 'w') as f:
    f.write('COLS: ' + str(list(df.columns)) + '\n')
    f.write('SHAPE: ' + str(df.shape) + '\n')
    f.write('NAN: ' + str(df.isna().sum().sum()) + '\n')
    f.write('DTYPES: ' + str(df.dtypes.to_dict()) + '\n')
    f.write('HEAD:\n' + df.to_string() + '\n')

with open(result_file) as f:
    print(f.read())
