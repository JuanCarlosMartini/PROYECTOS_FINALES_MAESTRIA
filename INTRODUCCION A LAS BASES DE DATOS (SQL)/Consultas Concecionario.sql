use concecionario;

select *
from tipo_empleado
where tipo_empleado = 'vendedor';

#consulta A
select e.dpi_empleado as 'dpi empleado', te.tipo_empleado as 'tipo empleado',count(s.id_tipo_servicio) as 'vehiculos vendidos',sum(s.costo_servicio) as 'monto venta'
from (empleado as e left outer join servicio as s on e.dpi_empleado = s.dpi_empleado), tipo_empleado as te
where e.id_tipo_empleado = te.id_tipo_empleado and e.id_tipo_empleado = "2"
group by e.dpi_empleado;

#consulta B
select e.dpi_empleado as 'DPI Empleado', s.fecha_servicio as 'Fecha de Servicio'
from (empleado as e join servicio as s on e.dpi_empleado = s.dpi_empleado)
where s.id_tipo_servicio = 2 and s.id_tipo_vehiculo = 1 and s.fecha_servicio > '2017-10-24' and s.fecha_servicio < '2017-10-31' and
	(select count(s.id_repuesto)
		where s.id_tipo_servicio = 2
		group by s.chasis_vehiculo
	) >=1
order by e.dpi_empleado;

#consulta C
select te.tipo_empleado, e.salario, s.nombre
from empleado as e, tipo_empleado as te, sucursal as s
where e.id_tipo_empleado = te.id_tipo_empleado and e.id_sucursal = s.id_sucursal
group by e.id_tipo_empleado, e.id_sucursal
order by e.id_sucursal;

#consulta d
select count(v.id_tipo_vehiculo) as cantidad, tv.tipo_vehiculo, v.marca, v.linea, v.modelo, s.nombre
from vehiculo as v,sucursal as s,tipo_vehiculo as tv
where v.id_sucursal = s.id_sucursal and v.id_tipo_vehiculo = tv.id_tipo_vehiculo
group by v.id_tipo_vehiculo, v.marca,v.linea,v.modelo,s.nombre
order by s.nombre,tv.tipo_vehiculo;


#consulta top 5 vehiculos mejor vendidos
select marca, linea, count(id_tipo_servicio) as 'cantidad vendida'
from (vehiculo as v join servicio as s on v.chasis = s.chasis_vehiculo)
where s.id_tipo_servicio = 1
group by v.marca, v.linea
order by count(id_tipo_servicio) DESC
limit 5;

select marca, sum(costo_servicio) as 'costo servicio'
from (vehiculo as v join servicio as s on v.chasis = s.chasis_vehiculo)
where s.id_tipo_servicio = 2
group by v.marca
order by count(id_tipo_servicio) DESC
limit 5
