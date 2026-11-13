let camera, scene, renderer, loader, light1, light2, controls, model;

init();
animate();

function startStreaming(url) {
    if (loader) {
        loader.pause();
    }
    if (model) {
        scene.remove(model);
    }

    const qosSelect = document.getElementById('qosMode');
    const netSelect = document.getElementById('netProfile');
    const qosEnabled = qosSelect ? (qosSelect.value === 'qos') : true;
    const simulatedKbps = netSelect ? Number(netSelect.value) : 0;

    model = new Model(url);
    scene.add(model);

    loader = new Loader(url, 1024, 20, {
        qosEnabled: qosEnabled,
        simulatedKbps: simulatedKbps
    });

    const btn = document.getElementById('btnPause');
    if (btn) {
        btn.innerHTML = "⏸️ Pause";
        btn.style.background = "#2980b9";
    }

    loader.start(function(elements) {
        for (let element of elements) {
            if (element !== undefined) {
                try {
                    model.manageElement(element);
                } catch(e) {
                    if (e.type === "custom") {
                        const errEl = document.getElementById('errormessage');
                        errEl.style.display = "block";
                        errEl.innerHTML = e;
                    } else {
                        throw e;
                    }
                }
            }
        }
    });
}

function init() {

    let url = (document.URL.split('?')[1] || "example/suzanne.obja");

    camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.0001, 1000);
    camera.position.z = 3;

    scene = new THREE.Scene();

    light1 = new THREE.AmbientLight(0x999999);
    scene.add(light1);

    light2 = new THREE.DirectionalLight(0xffffff, 1.0);
    light2.position.set(0.0, 1.0, 0.0);
    scene.add(light2);

    renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    window.addEventListener('resize', onWindowResize, false);
    controls = new THREE.OrbitControls(camera, renderer.domElement);

    startStreaming(url);
}

function restartStreaming() {
    let url = (document.URL.split('?')[1] || "example/suzanne.obja");
    startStreaming(url);
}

function onStreamingSettingChange() {
    if (!loader) return;
    const qosSelect = document.getElementById('qosMode');
    const netSelect = document.getElementById('netProfile');
    if (qosSelect) loader.setQosEnabled(qosSelect.value === 'qos');
    if (netSelect) loader.setSimulatedKbps(Number(netSelect.value));
}

function togglePause() {
    let btn = document.getElementById('btnPause');
    if (!loader.paused) {
        loader.pause();
        if (btn) {
            btn.innerHTML = "▶️ Reprendre";
            btn.style.background = "#e67e22";
        }
    } else {
        loader.resume();
        if (btn) {
            btn.innerHTML = "⏸️ Pause";
            btn.style.background = "#2980b9";
        }
    }
}

function takeScreenshot() {
    renderer.render(scene, camera);
    let dataUrl = renderer.domElement.toDataURL('image/png');
    let a = document.createElement('a');
    let rawParam = document.URL.split('?')[1] || "modele";
    let modelName = rawParam.split('/').pop().replace('.obja', '').replace('.obj', '');
    let pct = Math.floor(100 * loader.percentage());
    a.download = `${modelName}_${pct}pct.png`;
    a.href = dataUrl;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

function animate() {

    requestAnimationFrame(animate);
    controls.update();
    document.getElementById('progressbar').value = Math.floor(100 * loader.percentage()) || 0;
    document.getElementById('percentage').innerHTML = Math.floor(100 * loader.percentage()) / 100 + "%";
    const qosEl = document.getElementById('qosStatus');
    if (qosEl && loader) {
        const kbps = loader.estimatedBps ? (loader.estimatedBps / 1024).toFixed(1) : "—";
        const chunkKb = (loader.lastChunkSize / 1024).toFixed(1);
        const sec = (loader.elapsedMs / 1000).toFixed(1);
        qosEl.textContent = `Lot: ${chunkKb} Ko | Débit: ${kbps} Ko/s | ${loader.requestCount} req (${sec}s)`;
    }
    renderer.render(scene, camera);

}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize( window.innerWidth, window.innerHeight );
}
