
from django.core.paginator import  Paginator

if __name__ == '__main__':
    item_list = ['john','paul','george','ringo']
    p = Paginator(item_list,2)
    print(p.count)
    print(p.page(2))
    print(p.page(2).object_list)
    print(p.page(2).number)
    print(p.num_pages)
    print(p.page(2).has_previous())
    print(p.page(2).previous_page_number())

    print(p.page(2).has_next())
    print(p.page(2).next_page_number())




