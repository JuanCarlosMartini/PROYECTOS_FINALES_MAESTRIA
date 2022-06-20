create database concesionario;

use concesionario;

#creando tabla sucursal
create table SUCURSAL (
	id_sucursal int not null,
    nombre varchar (45) not null,
    direccion varchar (45) not null,
    primary key (id_sucursal)
);

#creando tabla cliente
create table CLIENTE (
	dpi_cliente integer not null,
    primer_nombre varchar(20) not null,
    segundo_nombre varchar(20) not null,
    primer_apellido varchar(20) not null,
    segundo_apellido varchar(20) not null,
    apellido_casada varchar (20),
    telefono integer not null,
    direccion varchar(60),
    primary key (dpi_cliente)
);

#creando tabla tipo empleado
create table TIPO_EMPLEADO (
	id_tipo_empleado int not null,
    tipo_empleado varchar(45) not null,
    primary key(id_tipo_empleado)
);

#creando tabla tipo de servicio
create table TIPO_SERVICIO (
	id_tipo_servicio int not null,
    tipo_servicio varchar(45) not null,
    primary key(id_tipo_servicio)
);

#creando tabla tipo de vehiculo
create table TIPO_VEHICULO (
	id_tipo_vehiculo int not null,
    tipo_vehiculo varchar(45) not null,
    primary key(id_tipo_vehiculo)
);

#creando tabla estado del vehiculo
create table ESTADO_VEHICULO (
	id_estado_vehiculo int not null,
    estado_vehiculo varchar(45) not null,
    primary key(id_estado_vehiculo)
);

#creando tabla tipo de repuesto
create table TIPO_REPUESTO (
	id_tipo_repuesto int not null,
    tipo_repuesto varchar(45) not null,
    primary key(id_tipo_repuesto)
);

#creando tabla empleado
create table EMPLEADO (
	dpi_empleado integer not null,
    primer_nombre varchar(45) not null,
    segundo_nombre varchar(45) not null,
    primer_apellido varchar(45) not null,
    segundo_apellido varchar(45) not null,
    apellido_casada varchar (45),
    fecha_contratacion date not null,
    salario double not null,
    id_tipo_empleado int not null,
    id_sucursal int not null,
    primary key(dpi_empleado),
    foreign key(id_tipo_empleado) references TIPO_EMPLEADO(id_tipo_empleado),
    foreign key(id_sucursal) references SUCURSAL(id_sucursal)
);

#creando tabla vehiculo
create table VEHICULO (
	chasis varchar(45) not null,
    placa varchar(20),
    marca varchar(45) not null,
    modelo int not null,
    linea varchar(45),
    color varchar(45) not null,
    km_recorrido int not null,
    id_tipo_vehiculo int not null,
    id_estado_vehiculo int not null,
    id_sucursal int not null,
    primary key(chasis),
    foreign key(id_tipo_vehiculo) references TIPO_VEHICULO(id_tipo_vehiculo),
    foreign key(id_estado_vehiculo) references ESTADO_VEHICULO(id_estado_vehiculo),
    foreign key(id_sucursal) references SUCURSAL(id_sucursal)
);

#creando tabla repuesto
create table REPUESTO (
	id_repuesto int not null,
    marca varchar(45) not null,
    modelo int not null,
    linea varchar(45),
    cantidad int not null,
    precio double,
    id_tipo_repuesto int not null,
    id_sucursal int not null,
    primary key(id_repuesto),
    foreign key(id_tipo_repuesto) references TIPO_REPUESTO(id_tipo_repuesto),
    foreign key(id_sucursal) references SUCURSAL(id_sucursal)
);

#creando tabla servicio
create table SERVICIO (
	id_servicio int not null auto_increment,
	fecha_servicio date not null,
    costo_servicio double not null,
    horas_servicio double,
    id_tipo_servicio int not null,
    chasis_vehiculo varchar(45),
    id_tipo_vehiculo int,
    id_estado_vehiculo int,
    dpi_empleado int not null,
    id_tipo_empleado int not null,
    id_sucursal int not null,
    dpi_cliente int not null,
    id_repuesto int,
    id_tipo_repuesto int,
    primary key(id_servicio),
    foreign key(id_tipo_servicio) references TIPO_SERVICIO(id_tipo_servicio),
    foreign key(chasis_vehiculo) references VEHICULO(chasis),
    foreign key(id_tipo_vehiculo) references TIPO_VEHICULO(id_tipo_vehiculo),
    foreign key(id_estado_vehiculo) references ESTADO_VEHICULO(id_estado_vehiculo),
    foreign key(dpi_empleado) references EMPLEADO(dpi_empleado),
    foreign key(id_tipo_empleado) references TIPO_EMPLEADO(id_tipo_empleado),
    foreign key(id_sucursal) references SUCURSAL(id_sucursal),
    foreign key(dpi_cliente) references CLIENTE(dpi_cliente),
    foreign key(id_repuesto) references REPUESTO(id_repuesto),
    foreign key(id_tipo_repuesto) references TIPO_REPUESTO(id_tipo_repuesto)
);
