# broomba #  

## Acknowledgements ##  
First and foremost, huge thanks to the Dustforce community Discord, both for exceptional technical knowledge and having no shortage of patience with me at times.  
Credit for the name goes to Skyhawk. Came up with it almost instantly. Absolute mad genius.  
Thanks to Alexspeedy for cutting off my hours-long debugging session by telling me about unbuffer.  
Thanks to msg555 for the fix for the game not taking input without being focused. Thanks also to msg for dustmod, without which this would have been much more difficult.  
Special thanks to Skyhawk for talking with me for literally hours, helping me work out how to approach getting input and output to and from Dustforce. 

### How To Use ###  
#### Dependencies ####
dustmod - I mean, obviously.  
xdotool - be my guest if you'd like to port the input mechanisms to launch autohotkey scripts or something, but xdotool is quite effective for sending keystrokes directly to windows. 
Note that Dustmod doesn't actually allow you to send inputs to it while unfocused. 
Well, it does, but they're buffered until it's focused again. This can be fixed; if you want to know how, ask msg.  
pytorch - and also, of course, python if you don't have it.  
unbuffer - necessary for working with python in this sort of context; clib buffers by default and
terminals override this behavior. unbuffer fixes that for a given output stream.  
I think that's everything, but if you're trying to get it working and I missed something, let me know please!  

#### How to run ####
Before you start, make sure to go to your Dustforce directory and put asciiforce.cpp in user/script_src
so that you can load it up. 
Next, run "python3 broomba.py" (or just python if you've got that alias set up). 
From here, go to any map with this script loaded. You can either do this by making a custom 
map with a script, or using editor everywhere from the dustmod options. Once it compiles successfully,
you can press return (out of editor mode!) and it'll automatically restart the level and begin. Don't worry
about the fact that it presses r so many times - that's intentional! It doesn't actually affect anything,
and xdotool very occasionally drops inputs.  

Oh, and don't expect fantastic results. But feel free to see if you can make them better, though the
quality of my code isn't exactly super high for this. Sorry.  

#### If you find an issue or have any questions at all ####
Let me know! Being able to help with questions is frankly quite fun, and I don't expect to be
overwhelmed by inquiries. @ me or DM me from the Dustforce discord.  