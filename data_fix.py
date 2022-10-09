import pickle as pk

pen_pp = ["pendulum_ppo1665328960","pendulum_ppo1665328964","pendulum_ppo1665328968","pendulum_ppo1665330666","pendulum_ppo1665330669"]
for j in pen_pp:
    with open(j,"rb") as f:
        hm=pk.load(f)

    evz=hm['evaluations']
    a=[[sum([x['reward'] for x in j]) for j in i] for i in evz]
    hm['evaluations']=a
    with open(j+"fix","wb") as f:
        pk.dump(hm,f)