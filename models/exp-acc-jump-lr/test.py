"""
import os
root_path='/home/itrocks/Data/Fashion-Upper-DF/'

uppers = ['Anorak', 'Blazer', 'Blouse', 'Bomber', 'Button-Down', 'Cardigan', 'Halter', 'Henley',
          'Hoodie', 'Jacket', 'Jersey', 'Parka', 'Peacoat', 'Poncho', 'Sweater', 'Tank', 'Tee', 'Top', 'Turtleneck']


for upper in uppers:
  files = os.listdir(root_path + upper)

  for file in files:
    print(file)

"""

a = [[1, 2, 3]]
print(a)

print(','.join(str(e) for e in a[0]))