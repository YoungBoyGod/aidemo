# class Node:
#     def __init__(self, value):
#         self.value = value
#         self.next = None

# class LinkedList:
#     def __init__(self):
#         self.head = None # 链表头
    
#     def insert(self, value): 
#         new_node=Node(value)
#         if not self.head:
#             self.head=new_node
#             return 
#         last_node=self.head
#         while last_node.next:
#             last_node=last_node.next
#         last_node.next=new_node 

#     def delete(self,key):
#         current_node=self.head
#         if current_node and current_node.value==key
#             self.head=current_node.next
#             current_node=None
#             return
#         prev=None
#         while current_node and current_node.value !=key:
#             prev=current_node 
#             current_node=current_node.next
#         if current_node is None:
#             return 
#         prev.next=current_node.next

#     def traverse(self):
#         current_node=self.head
#         while current_node:
#             print(current_node.value,end="->")
#             current_node=current_node.next
#         print("None")

#     def reverse(self):
#         prev=None
#         current_node=self.head
#         while current_node:
#             next_node=current_node.next
#             current_node.next=prev
#             prev=current_node
#             current_node=next_node
#         self.head=prev
#         return self.head          

class  DNode:
    def __init__(self,data) -> None:
        self.data=data
        self.next=None
        self.prev=None


class DLinkedList:
    def __init__(self):
        self.head=None

    def  insert(self,data):
        new_node=DNode(data)
        if not self.head:
            self.head=new_node
            return 
        last=self.head
        while last.next:
            last=last.next
        last.nesxt=new_node
        new_node.prev=last

    def delete(self,key):
        current=self.head
        # bianli
        while current and current.data!=key:
            current=current.next
        #  meizhaoidao
        if current is None:
            return 
        # zhaodao
        if current.prev:
            current.prev.next=current.next
        if current.next:
            current.next.prev=current.prev
        if current ==self.head:
            self.head=current.next
        current=None

    def traverse(self):
        current=self.head
        while current:
            print(current.data,end="->")
            current=current.next
        print("None")
    
    def reverse(self):
        current=self.head
        while current:
            prev=current.prev
            next=current.next
            current.prev=next
            current.next=prev
            current=current.prev
        self.head=prev.prev


    def reverse(self):
        current=self.head
        while current:
            prev=current.prev
            next=current.next
            current.prev=next
            current.next=prev
            current=current.prev
        self.head =prev.prev