with import <nixpkgs> { };

let
  pythonPackages = python311Packages; # Change to Python 3.11
in pkgs.mkShell rec {
  venvDir = "./.venv";
  buildInputs = [
    pkgs.stdenv.cc.cc.lib
    pythonPackages.ipykernel
    pythonPackages.jupyterlab
    pythonPackages.pyzmq    # Adding pyzmq explicitly
    pythonPackages.venvShellHook
    pythonPackages.pip
  ];
  
  shellHook  = ''
    # fixes libstdc++ issues and libgl.so issues
    LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib/:/run/opengl-driver/lib/
  '';
}