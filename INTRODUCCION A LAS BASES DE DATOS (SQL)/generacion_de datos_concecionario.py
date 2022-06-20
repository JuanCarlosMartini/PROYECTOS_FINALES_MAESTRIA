import random
import string
import datetime
#for i in range(41):
#    print( "("+str(i)+",'ncliente"+str(i)+"','sncliente"+str(i)+"','acliente"+str(i)+"','sacliente"+str(i)+"','',"+''.join(random.choices(string.digits, k=8))+",'direccion"+str(i)+"',"+''.join(random.choices(string.digits, k=5))+")")
marcas = {"echo":"toyota","prado":"toyota","prius":"toyota","tercel":"toyota","tacoma":"toyota","mazda":"","lancer":"mitsubishi","subaru":"","sportage":"kia","sorento":"kia","civic":"honda","acura":"honda","ranger":"ford","GT":"ford","festival":"ford"}
precios = [150,250,300,500,750,1000,2000]
#for i in range(41):
#    modelo,marca = random.choice(list(marcas.items()))
#print("("+str(i)+",'" + marca +"',"+str(random.choice(range(2000,2022)))+ ",'"+modelo+"',"+str(random.choice(precios))+","+str(random.choice([1,2]))+"),")


def fechas():
    start_date = datetime.date(2015, 1, 1)
    end_date = datetime.date(2021, 10, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_number_of_days = random.randrange(days_between_dates)
    random_date = start_date + datetime.timedelta(days=random_number_of_days)
    return random_date
#ventas
#for i in range(26):
#    print("("+str(i)+",'"+str(fechas())+"',"+str(random.choice(range(15000,100000,5000)))+",,1,"+str(i)+",1,idestadovehi,"+str(random.choice([6,7,8,9]))+",2,sucursal,"+str(i)+"),")

#servicios
for i in range(26,51):
    print("("+str(i)+",'"+str(fechas())+"',"+str(random.choice(range(1000,10000,500)))+","+str(random.choice(range(5,16,1)))+",2,"+str(i-25)+",1,idestadovehi,"+str(random.choice([1,2,3,4,5]))+",3,sucursal,"+str(i-25)+",idrep,idtiporep),")



    
