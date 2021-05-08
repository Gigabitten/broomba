// Script for use with dustmod.
// Prints an ascii display of the given width and height around the camera to the console.
// Charming.

const int WIDTH = 30;
const int HEIGHT = 20;

class script {
  camera@ cam;
  scene@ g;
  controllable@ player;
  
  script() {
    @cam = get_active_camera();
    @g = get_scene();
  }

  uint8 side_type(uint8 side) {
    if(side == 0) return ' '[0];
    else if(side >= 1 and side <= 5) return '~'[0];
    else return '^'[0];
  }

  void step(int entities) {
    array<string> tiles;
    tileinfo@ tile;
    tilefilth@ filth;
    for(int i = 0; i < HEIGHT; i++) {
      tiles.insertLast(" ");
        for(int j = 0; j < WIDTH; j++) {
          int x = int((cam.x() / 48) + j - int(WIDTH / 2));
          int y = int((cam.y() / 48) + i - int(HEIGHT / 2));          
          @tile = g.get_tile(x, y);
          tiles[i]+= ' ';
          if(tile.solid()) tiles[i][j] = '#'[0];
          if(tile.is_dustblock()) tiles[i][j] = '~'[0];
      }      
    }
    for(int i = 0; i < HEIGHT; i++) {
        for(int j = 0; j < WIDTH; j++) {
          int x = int((cam.x() / 48) + j - int(WIDTH / 2));
          int y = int((cam.y() / 48) + i - int(HEIGHT / 2));          
          @filth = g.get_tile_filth(x, y);
          uint8 top = side_type(filth.top());
          uint8 bottom = side_type(filth.bottom());
          uint8 left = side_type(filth.left());
          uint8 right = side_type(filth.right());
          if(top != ' '[0] and i > 0) tiles[i-1][j] = top;
          if(bottom != ' '[0] and i < 19) tiles[i+1][j] = bottom;
          if(left != ' '[0] and j > 0) tiles[i][j-1] = left;
          if(right != ' '[0] and j < 29) tiles[i][j+1] = right;
      }      
    }
    int enemyCount = g.get_entity_collision(cam.y() - ((HEIGHT / 2) * 48),
                                            cam.y() + ((HEIGHT / 2) * 48),
                                            cam.x() - ((WIDTH / 2) * 48),
                                            cam.x() + ((WIDTH / 2) * 48),
                                            1);
    entity@ enemy;
    for(int i = 0; i < enemyCount; i++) {
      @enemy = g.get_entity_collision_index(i);
      controllable@ con = enemy.as_controllable();
      if(con.life() > 0) {
        int y = int(((enemy.y() - cam.y()) / 48) + (HEIGHT / 2) - 1);
        int x = int(((enemy.x() - cam.x()) / 48) + (WIDTH / 2) - 1);
        if(x >= 0 and x < WIDTH and y >= 0 and y < HEIGHT) tiles[y][x] = '@'[0];
      }
    }
    @player = controller_controllable(0);
    if(@player != null) {
      int y = int(((player.y() - cam.y()) / 48) + (HEIGHT / 2));
      int x = int(((player.x() - cam.x()) / 48) + (WIDTH / 2));    
      tiles[y-1][x] = 'P'[0];
    }
    
    for(int i = 0; i < 5; i++) puts("");
    for(uint i = 0; i < tiles.length(); i++) puts(tiles[i]);
  }
}

