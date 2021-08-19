select * from res_partner
--where length(document_number)>10

select * from res_partner where id = 238512

select ai.id "InvoiceNo",
ail.product_id "StockCode",  
ail.name "Description",
pc.name "Category",
ppc.name "ParentCategory",
ail.quantity "Quantity", 
ai.date_invoice "InvoiceDate", 
ail.price_unit "UnitPrice",
--ail.price_subtotal,
rp.id "CustomerID",
rp.city "City"
from account_invoice_line ail 
join account_invoice ai on (ail.invoice_id=ai.id)
join res_partner rp on (ai.partner_id=rp.id)
left join product_product pp  on (ail.product_id=pp.id)
left join product_template pt  on (pp.product_tmpl_id=pt.id)
left join product_category pc on (pt.categ_id=pc.id)
left join product_category ppc on (pc.parent_id = ppc.id)
where ai.partner_id!=1
and ail.quantity > 0
and ail.price_unit between 0 and 250000
order by price_subtotal desc
--order by 4

select * from account_invoice_line limit 10

select * from sale_order_line limit 10
select count(*) from sale_order_line 
select * from res_country where id = 11

select * from account_invoice limit 10

select * from res_partner limit 10

select count (*) from res_partner

select count (*) from account_invoice

select count (*) from account_invoice_line where 

select count(*) from product_product
select count(*) from product_template

select
