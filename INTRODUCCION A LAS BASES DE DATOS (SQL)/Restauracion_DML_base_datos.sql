use concesionario;

#ingresando sucursales
insert into sucursal
	values
    (1,'central','guate'),
    (2,'norte','peten'),
    (3,'sur','escuintla');

#ingresando tipos de vehiculo
insert into tipo_vehiculo
	values 
    (1,'automovil'),
	(2,'motocicleta');

#ingresando estado de vehiculos
insert into estado_vehiculo
	values
    (1,'nuevo'),
    (2,'importado'),
    (3,'usado');

#ingresando automoviles
insert into vehiculo
	values 
    (1,'1a','toyota',2000,'echo','verde',50000,1,3,1),
    (2,'2a','toyota',2010,'echo','azul',30000,1,3,1),
    (3,'3a','mitsubishi',2018,'lancer','negro',5000,1,2,1),
    (4,'','mitsubishi',2021,'','verde',0,1,1,2),
    (5,'5a','toyota',2018,'prado','negro',15000,1,2,2),
    (6,'','toyota',2021,'prado','rojo',0,1,1,3),
    (7,'7a','subaru',2006,'','verde',45000,1,3,3),
    (8,'8a','subaru',2010,'','rojo',25000,1,2,1),
    (9,'9a','kia',2009,'sportage','blanca',35000,1,2,2),
    (10,'','kia',2021,'','negro',0,1,1,3),
    (11,'11a','kia',2015,'sport','azul',35000,1,2,2),
    (12,'12a','toyota',2014,'prado','blanco',60000,1,3,2),
    (13,'13a','toyota',2018,'','blanco',40000,1,2,1),
    (14,'14a','mazda',2014,'','azul',60000,1,3,1),
    (15,'15a','mazda',2020,'','negro',10000,1,2,3),
    (16,'','mazda',2021,'','azul',0,1,1,1),
    (17,'','honda',2021,'','negro',0,1,1,2),
    (18,'18a','honda',2010,'civic','rojo',60000,1,3,1),
    (19,'19a','honda',2015,'civic','negro',40000,1,2,3),
    (20,'20a','honda',2017,'acura','amarillo',30000,1,2,1),
    (21,'21a','kia',2021,'sorento','negro',5000,1,2,2),
    (22,'','kia',2022,'sorento','blanca',0,1,1,2),
    (23,'23a','ford',2015,'ranger','negro',40000,1,3,2),
    (24,'24a','ford',2022,'GT','negro',0,1,1,3),
    (25,'25a','ford',2018,'festival','rojo',22000,1,2,1),
    (26,'26a','subaru',2009,'','blanco',80000,1,3,3),
    (27,'27a','subaru',2011,'','azul',65000,1,3,1),
    (28,'28a','toyota',2020,'prius','negro',8000,1,2,1),
    (29,'29a','toyota',2006,'tercel','dorado',65000,1,3,3),
    (30,'','toyota',2022,'tacoma','negro',0,1,1,3),
    (41,'','toyota',2022,'tacoma','blanco',0,1,1,3);

#ingresando motocicletas
insert into vehiculo
	values 
    (31,'31a','yamaha',2015,'','verde',12000,2,2,1),
    (32,'32a','yamaha',2019,'','azul',5000,2,3,1),
    (33,'','yamaha',2022,'','negro',0,2,1,2),
    (34,'34a','honda',2012,'','negro',20000,2,3,2),
    (35,'35a','honda',2018,'','azul',10000,2,2,3),
    (36,'','honda',2022,'','rojo',0,2,1,2),
    (37,'37a','suzuki',2006,'','azul',18000,2,3,1),
    (38,'38a','suzuki',2019,'','rojo',3000,2,2,1),
    (39,'','suzuki',2022,'','rojo',0,2,1,3),
    (40,'','kawasaki',2021,'','blancao',3000,2,2,3);

#ingresando tipo empleado
insert into tipo_empleado
	values
    (1,'administrativo'),
    (2,'vendedor'),
    (3,'mecanico');

#ingresando empleados
insert into empleado
	values
    (1,'Raul','Esteban','Garcia','Rodas','','2006-06-05',4000,3,1),
    (2,'Jose','Alberto','Ramon','Perez','','2007-06-05',4000,3,2),
    (3,'Berny','Leonel','Reyes','Reyes','','2008-06-05',4000,3,1),
    (4,'Victor','Leonel','Morales','Reyes','','2008-06-05',4000,3,2),
    (5,'Joaquin','Antonio','Preya','Bonan','','2008-07-05',4000,3,3),
    (6,'Julio','Esteban','Mora','Rodas','','2004-06-05',5000,2,1),
    (7,'Maria','Jose','Perea','Perez','','2005-06-05',5000,2,2),
    (8,'Karla','Andrea','Aguilar','Reyes','Rosario','2005-06-05',5000,2,3),
    (9,'Javier','Antonio','Xi','Caal','','2005-06-05',5000,2,1),
    (10,'Meli','Karina','Rodas','Perea','','2004-07-12',6000,1,1),
    (11,'Walter','Leonel','Alvizures','Caal','','2007-04-05',6000,1,2),
    (12,'Carlos','Antonio','Poc','Yax','','2007-03-05',6000,1,3),
    (13,'Cesar','Antonio','R','C','','2009-03-05',4000,2,1); #nuevo empleado
    
#ingresando clientes
insert into cliente
	values
	(1,'ncliente1','sncliente1','acliente1','sacliente1','',68300987,'direccion1'),
	(2,'ncliente2','sncliente2','acliente2','sacliente2','',55041034,'direccion2'),
	(3,'ncliente3','sncliente3','acliente3','sacliente3','',08321399,'direccion3'),
	(4,'ncliente4','sncliente4','acliente4','sacliente4','',84868328,'direccion4'),
	(5,'ncliente5','sncliente5','acliente5','sacliente5','',54962153,'direccion5'),
	(6,'ncliente6','sncliente6','acliente6','sacliente6','',41769919,'direccion6'),
	(7,'ncliente7','sncliente7','acliente7','sacliente7','',46479542,'direccion7'),
	(8,'ncliente8','sncliente8','acliente8','sacliente8','',38073091,'direccion8'),
	(9,'ncliente9','sncliente9','acliente9','sacliente9','',94601214,'direccion9'),
	(10,'ncliente10','sncliente10','acliente10','sacliente10','',75580762,'direccion10'),
	(11,'ncliente11','sncliente11','acliente11','sacliente11','',75472846,'direccion11'),
	(12,'ncliente12','sncliente12','acliente12','sacliente12','',89758355,'direccion12'),
	(13,'ncliente13','sncliente13','acliente13','sacliente13','',00605219,'direccion13'),
	(14,'ncliente14','sncliente14','acliente14','sacliente14','',64295126,'direccion14'),
	(15,'ncliente15','sncliente15','acliente15','sacliente15','',35293148,'direccion15'),
	(16,'ncliente16','sncliente16','acliente16','sacliente16','',58080971,'direccion16'),
	(17,'ncliente17','sncliente17','acliente17','sacliente17','',15449568,'direccion17'),
	(18,'ncliente18','sncliente18','acliente18','sacliente18','',54244352,'direccion18'),
	(19,'ncliente19','sncliente19','acliente19','sacliente19','',34326399,'direccion19'),
	(20,'ncliente20','sncliente20','acliente20','sacliente20','',07323572,'direccion20'),
	(21,'ncliente21','sncliente21','acliente21','sacliente21','',54441440,'direccion21'),
	(22,'ncliente22','sncliente22','acliente22','sacliente22','',12523061,'direccion22'),
	(23,'ncliente23','sncliente23','acliente23','sacliente23','',13508847,'direccion23'),
	(24,'ncliente24','sncliente24','acliente24','sacliente24','',35079802,'direccion24'),
	(25,'ncliente25','sncliente25','acliente25','sacliente25','',01572251,'direccion25'),
	(26,'ncliente26','sncliente26','acliente26','sacliente26','',14273761,'direccion26'),
	(27,'ncliente27','sncliente27','acliente27','sacliente27','',27593391,'direccion27'),
	(28,'ncliente28','sncliente28','acliente28','sacliente28','',76746711,'direccion28'),
	(29,'ncliente29','sncliente29','acliente29','sacliente29','',24479028,'direccion29'),
	(30,'ncliente30','sncliente30','acliente30','sacliente30','',16689107,'direccion30'),
	(31,'ncliente31','sncliente31','acliente31','sacliente31','',00162445,'direccion31'),
	(32,'ncliente32','sncliente32','acliente32','sacliente32','',09434818,'direccion32'),
	(33,'ncliente33','sncliente33','acliente33','sacliente33','',89930144,'direccion33'),
	(34,'ncliente34','sncliente34','acliente34','sacliente34','',96469436,'direccion34'),
	(35,'ncliente35','sncliente35','acliente35','sacliente35','',49634642,'direccion35'),
	(36,'ncliente36','sncliente36','acliente36','sacliente36','',53577442,'direccion36'),
	(37,'ncliente37','sncliente37','acliente37','sacliente37','',75687266,'direccion37'),
	(38,'ncliente38','sncliente38','acliente38','sacliente38','',15944268,'direccion38'),
	(39,'ncliente39','sncliente39','acliente39','sacliente39','',34805492,'direccion39'),
	(40,'ncliente40','sncliente40','acliente40','sacliente40','',75142478,'direccion40');

#ingresando tipo de repuesto
insert into tipo_repuesto
	values
    (1,'automovil'),
    (2,'motocicleta');

#ingresando repuestos
insert into repuesto
	values
    (1,'honda',2014,'civic',150,10,1,1),
	(2,'ford',2002,'festival',2000,3,2,1),
	(3,'kia',2021,'sportage',250,5,1,3),
	(4,'mazda',2014,'',500,7,1,2),
	(5,'honda',2014,'civic',2000,10,2,1),
	(6,'toyota',2011,'tacoma',500,15,2,1),
	(7,'honda',2013,'acura',750,18,1,2),
	(8,'toyota',2009,'tacoma',150,25,2,3),
	(9,'ford',2005,'festival',2000,14,2,3),
	(10,'toyota',2018,'echo',2000,9,2,2),
	(11,'honda',2007,'civic',300,12,1,1),
	(12,'honda',2018,'civic',750,15,2,1),
	(13,'toyota',2007,'tacoma',500,16,2,2),
	(14,'subaru',2012,'',250,10,2,3),
	(15,'honda',2011,'acura',1000,18,2,1),
	(16,'honda',2015,'civic',500,22,2,2),
	(17,'kia',2011,'sorento',250,24,1,3),
	(18,'ford',2014,'ranger',500,23,2,1),
	(19,'kia',2002,'sorento',300,11,2,2),
	(20,'mazda',2019,'',1000,32,2,1),
	(21,'kia',2013,'sportage',150,40,1,1),
	(22,'toyota',2020,'prado',1000,17,2,3),
	(23,'toyota',2010,'prius',150,19,2,3),
	(24,'kia',2018,'sorento',750,25,1,2),
	(25,'mazda',2013,'',300,30,2,2),
	(26,'ford',2004,'GT',250,38,2,1),
	(27,'toyota',2020,'echo',2000,40,2,1),
	(28,'subaru',2018,'',750,15,2,3),
	(29,'kia',2015,'sportage',500,17,2,1),
	(30,'toyota',2021,'prado',150,19,2,1),
	(31,'kia',2018,'sportage',150,26,1,3),
	(32,'toyota',2021,'prado',150,27,2,3),
	(33,'subaru',2013,'',750,28,2,2),
	(34,'mitsubishi',2012,'lancer',300,19,2,2),
	(35,'toyota',2016,'prius',500,16,2,2),
	(36,'ford',2012,'festival',2000,30,1,1),
	(37,'toyota',2013,'echo',300,31,2,1),
	(38,'honda',2016,'civic',750,24,1,3),
	(39,'mazda',2001,'',500,28,2,3),
	(40,'ford',2001,'festival',2000,29,1,2),
    (41,'ford',2001,'festival',2000,29,1,1),
    (42,'ford',2001,'festival',2000,29,1,3);

#ingresando tipo de servicio prestado
insert into tipo_servicio
	values
    (1,'venta'),
    (2,'servicio mecanico');
    
#ingresando ventas
INSERT INTO servicio (fecha_servicio,costo_servicio,id_tipo_servicio,chasis_vehiculo,id_tipo_vehiculo,id_estado_vehiculo,dpi_empleado,id_tipo_empleado,id_sucursal,dpi_cliente)
	values
	('2017-10-16',70000,1,3,1,2,8,2,3,3),
	('2016-10-15',40000,1,4,1,1,6,2,1,4),
	('2011-02-06',95000,1,5,1,2,6,2,1,5),
	('2019-07-08',50000,1,6,1,1,8,2,3,6),
	('2014-06-01',20000,1,7,1,3,9,2,1,7),
	('2011-10-31',60000,1,8,1,2,6,2,1,8),
	('2011-02-27',35000,1,9,1,2,7,2,2,9),
	('2012-09-21',15000,1,10,1,1,9,2,1,10),
	('2014-05-13',60000,1,11,1,2,9,2,1,11),
	('2010-05-28',95000,1,12,1,3,6,2,1,12),
	('2016-06-21',20000,1,13,1,2,6,2,1,13),
	('2019-06-24',45000,1,14,1,3,6,2,1,14),
	('2019-10-10',65000,1,15,1,2,9,2,1,15),
	('2020-04-28',75000,1,16,1,1,8,2,3,16),
	('2012-01-14',15000,1,17,1,1,7,2,2,17),
	('2012-12-06',35000,1,18,1,3,8,2,3,18),
	('2020-02-26',55000,1,19,1,2,9,2,1,19),
	('2019-11-01',60000,1,31,2,2,9,2,1,20),
	('2015-08-23',35000,1,32,2,3,9,2,1,21),
	('2013-10-10',40000,1,33,2,1,9,2,1,22),
	('2015-06-11',75000,1,34,2,3,8,2,3,23),
	('2016-02-19',80000,1,35,2,2,6,2,1,24),
	('2021-10-14',55000,1,40,2,2,9,2,1,25);
    
#ingresando reparaciones
INSERT INTO servicio 
	values
	(26,'2016-10-20',1000,5,2,1,1,3,2,3,2,1,1,1),
	(27,'2015-08-30',3000,7,2,2,1,3,4,3,2,2,2,2),
	(28,'2021-02-05',7000,10,2,3,1,2,4,3,2,3,3,1),
	(29,'2016-05-16',9500,11,2,4,1,1,2,3,2,4,4,1),
	(30,'2020-11-10',6000,5,2,5,1,2,1,3,1,5,5,2),
	(31,'2021-09-19',5500,14,2,6,1,1,1,3,1,6,6,2),
	(32,'2020-08-16',8000,6,2,7,1,3,5,3,3,7,7,1),
	(33,'2017-04-25',9000,13,2,8,1,2,2,3,2,8,8,2),
	(34,'2015-03-20',4000,5,2,9,1,2,1,3,1,9,9,2),
	(35,'2021-02-11',4500,12,2,10,1,1,5,3,3,10,10,2),
	(36,'2015-03-23',2000,14,2,11,1,2,5,3,3,11,11,1),
	(37,'2017-01-30',6500,8,2,12,1,3,2,3,2,12,12,2),
	(38,'2018-09-16',6000,8,2,13,1,2,3,3,1,13,13,2),
	(39,'2015-02-18',1500,5,2,14,1,3,4,3,2,14,14,2),
	(40,'2015-04-25',3500,7,2,15,1,2,4,3,2,15,15,2),
	(41,'2021-01-04',7500,8,2,16,1,1,5,3,3,16,16,2),
	(42,'2018-07-20',5000,14,2,17,1,1,5,3,3,17,17,1),
	(43,'2021-03-24',4500,13,2,18,1,3,1,3,1,18,18,2),
	(44,'2015-01-21',3500,15,2,19,1,2,3,3,1,19,19,2),
	(45,'2017-04-11',4500,9,2,31,2,2,3,3,1,20,20,2),
	(46,'2021-08-09',2500,15,2,32,2,3,1,3,1,21,21,1),
	(47,'2021-07-13',7000,14,2,33,2,1,3,3,1,22,22,2),
	(48,'2016-01-15',5500,8,2,34,2,3,2,3,2,23,23,2),	
	(49,'2017-10-27',4000,11,2,40,1,2,5,3,3,25,25,2);
    
#ingresando otras ventas
insert into servicio (fecha_servicio,costo_servicio,horas_servicio,id_tipo_servicio,chasis_vehiculo,id_tipo_vehiculo,id_estado_vehiculo,dpi_empleado,id_tipo_empleado,id_sucursal,dpi_cliente) 
    VALUES 
    ('2017-10-29',5000,6,2,35,1,2,4,3,2,24),
    ('2017-10-28',3500,8,2,36,1,3,1,3,1,21);