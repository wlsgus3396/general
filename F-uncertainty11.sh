for k in 1 2 3
do
    python3 main.py --execute 'vaal' --gpu 1 --K $k --iid 'False' --bn 'False' --momentum 'False' --budgetratio 0    
done


for k in 1 2 3
do
    python3 main.py --execute 'vaal' --gpu 1 --K $k --iid 'False' --bn 'True' --momentum 'False' --budgetratio 0.1    
done

for k in 1 2 3
do
    python3 main.py --execute 'vaal' --gpu 1 --K $k --iid 'False' --bn 'False' --momentum 'True' --budgetratio 0    
done


for k in 1 2 3
do
    python3 main.py --execute 'vaal' --gpu 1 --K $k --iid 'False' --bn 'True' --momentum 'True' --budgetratio 0.1    
done
