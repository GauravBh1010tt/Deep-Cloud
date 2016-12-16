err = [ error[i][1].tolist() for i in range(0,1000)]
it = [i  for i in range(0,1000)]
import matplotlib.pyplot as plt
plt.plot(it, err)
plt.xlabel('no. of iterations')
plt.ylabel('Cost')
plt.title('FARM ADS')
