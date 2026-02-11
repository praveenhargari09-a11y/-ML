#include<stdio.h>
int main()
{
    FILE*ptr;
    ptr=fopen("new.txt","w");
    fprintf(ptr,"hello");
    fclose(ptr);
    ptr=fopen("new.txt","a");
    fprintf(ptr,"world");
    fclose(ptr);
    ptr=fopen("new.txt","r");
    char s[20];
    fgets(s,20,ptr);
    printf("%s",s);
    fclose(ptr);
    rename("new.txt","file.txt");
    printf("/nsuccessfully renamed");
    return 0;
}