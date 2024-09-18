Set COUNTER=0
:x


echo %Counter%
if "%Counter%"=="4" (
    echo "END!"
) else (
    
    echo "DOING"
    start cmd.exe /c "python trainer.py %Counter%"
    
    set /A COUNTER+=1
    goto x
)