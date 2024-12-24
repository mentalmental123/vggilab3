function normalizeUV(value, min, max) {
    return (value - min) / (max - min);
}

function calculateNormalsAndTangents(vertices, indices, uvs) {
    const normals = new Float32Array(vertices.length).fill(0);
    const tangents = new Float32Array(vertices.length).fill(0);

    for (let i = 0; i < indices.length; i += 3) {
        const i1 = indices[i] * 3;
        const i2 = indices[i + 1] * 3;
        const i3 = indices[i + 2] * 3;

        const v1 = [vertices[i1], vertices[i1 + 1], vertices[i1 + 2]];
        const v2 = [vertices[i2], vertices[i2 + 1], vertices[i2 + 2]];
        const v3 = [vertices[i3], vertices[i3 + 1], vertices[i3 + 2]];

        const edge1 = [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]];
        const edge2 = [v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]];

        const uv1 = [uvs[indices[i] * 2], uvs[indices[i] * 2 + 1]];
        const uv2 = [uvs[indices[i + 1] * 2], uvs[indices[i + 1] * 2 + 1]];
        const uv3 = [uvs[indices[i + 2] * 2], uvs[indices[i + 2] * 2 + 1]];

        const deltaUV1 = [uv2[0] - uv1[0], uv2[1] - uv1[1]];
        const deltaUV2 = [uv3[0] - uv1[0], uv3[1] - uv1[1]];

        const f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV1[1] * deltaUV2[0]);

        const tangent = [
            f * (deltaUV2[1] * edge1[0] - deltaUV1[1] * edge2[0]),
            f * (deltaUV2[1] * edge1[1] - deltaUV1[1] * edge2[1]),
            f * (deltaUV2[1] * edge1[2] - deltaUV1[1] * edge2[2])
        ];

        const normal = m4.normalize(m4.cross(edge1, edge2, [0, 1, 0]), []);

        normals[i1] += normal[0];
        normals[i1 + 1] += normal[1];
        normals[i1 + 2] += normal[2];

        normals[i2] += normal[0];
        normals[i2 + 1] += normal[1];
        normals[i2 + 2] += normal[2];

        normals[i3] += normal[0];
        normals[i3 + 1] += normal[1];
        normals[i3 + 2] += normal[2];

        tangents[i1] += tangent[0];
        tangents[i1 + 1] += tangent[1];
        tangents[i1 + 2] += tangent[2];

        tangents[i2] += tangent[0];
        tangents[i2 + 1] += tangent[1];
        tangents[i2 + 2] += tangent[2];

        tangents[i3] += tangent[0];
        tangents[i3 + 1] += tangent[1];
        tangents[i3 + 2] += tangent[2];
    }

    for (let i = 0; i < normals.length; i += 3) {
        const nx = normals[i];
        const ny = normals[i + 1];
        const nz = normals[i + 2];

        const tx = tangents[i];
        const ty = tangents[i + 1];
        const tz = tangents[i + 2];

        const normalLength = Math.sqrt(nx * nx + ny * ny + nz * nz);
        const tangentLength = Math.sqrt(tx * tx + ty * ty + tz * tz);

        if (normalLength > 0) {
            normals[i] = nx / normalLength;
            normals[i + 1] = ny / normalLength;
            normals[i + 2] = nz / normalLength;
        }

        if (tangentLength > 0) {
            tangents[i] = tx / tangentLength;
            tangents[i + 1] = ty / tangentLength;
            tangents[i + 2] = tz / tangentLength;
        }
    }

    return { normals, tangents };
}


function createPoint(a, b, n, u, v) {
    let x = (a + b * Math.sin(n * u)) * Math.cos(u) - v * Math.sin(u);
    let y = (a + b * Math.sin(n * u)) * Math.sin(u) + v * Math.cos(u);
    let z = b * Math.cos(n * u);

    return [x, y, z]
}  

function generateSurface(a, b, n, uSteps, vSteps, vMin, vMax) {
    const vertices = [];
    let uvs = [];
    const indices = [];

    const uMin = 0.0, uMax = 2.0 * Math.PI;

    const du = (uMax - uMin) / uSteps;
    const dv = (vMax - vMin) / vSteps;

    let accomulatedU = 0.0;

    for (let i = 0; i <= uSteps; i++) {
        const u = uMin + i * du;

        {
            const current = createPoint(a, b, n, u, vMin);
            const prev = createPoint(a, b, n, u - du, 0);
            
            accomulatedU += m4.length(m4.subtractVectors(current, prev)) * 0.025;
        }
        
        let accomulatedV = 0.0;

        for (let j = 0; j <= vSteps; j++) {
            const v = vMin + j * dv;

            const current = createPoint(a, b, n, u, v);
            vertices.push(...current);

            const prev = createPoint(a, b, n, u, v - dv);
            accomulatedV += m4.length(m4.subtractVectors(current, prev)) * 0.25;
            uvs.push(accomulatedU, accomulatedV);
        }
    }

    for (let i = 0; i < uSteps; i++) {
        for (let j = 0; j < vSteps; j++) {
            const topLeft = i * (vSteps + 1) + j;
            const topRight = i * (vSteps + 1) + (j + 1);
            const bottomLeft = (i + 1) * (vSteps + 1) + j;
            const bottomRight = (i + 1) * (vSteps + 1) + (j + 1);

            indices.push(topLeft, bottomLeft, bottomRight);
            indices.push(topLeft, bottomRight, topRight);
        }
    }
    
    return { vertices, uvs, indices };
}

export default function Model(gl, shProgram) {
    this.iVertexBuffer = gl.createBuffer();
    this.iNormalBuffer = gl.createBuffer();
    this.iTangentBuffer = gl.createBuffer();
    this.iUVBuffer = gl.createBuffer();
    this.iIndexBuffer = gl.createBuffer();

    this.idTextureDiffuse = LoadTexture(gl, "./textures/diffuse.jpg");
    this.idTextureNormal = LoadTexture(gl, "./textures/normal.jpg");
    this.idTextureSpecular = LoadTexture(gl, "./textures/specular.jpg");

    this.count = 0;

    this.BufferData = function(vertices, normals, tangents, uvs, indices) {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iNormalBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, normals, gl.STATIC_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iTangentBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, tangents, gl.STATIC_DRAW);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iUVBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uvs), gl.STATIC_DRAW);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.iIndexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(indices), gl.STATIC_DRAW);

        this.count = indices.length;
    };

    this.Draw = function() {
        gl.bindBuffer(gl.ARRAY_BUFFER, this.iVertexBuffer);
        gl.vertexAttribPointer(shProgram.iAttribVertex, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribVertex);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iNormalBuffer);
        gl.vertexAttribPointer(shProgram.iAttribNormal, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribNormal);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iTangentBuffer);
        gl.vertexAttribPointer(shProgram.iAttribTangent, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribTangent);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.iUVBuffer);
        gl.vertexAttribPointer(shProgram.iAttribUV, 2, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(shProgram.iAttribUV);

        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.iIndexBuffer);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.idTextureDiffuse);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.idTextureNormal);

        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, this.idTextureSpecular);

        gl.drawElements(gl.TRIANGLES, this.count, gl.UNSIGNED_SHORT, 0);
    }

    this.CreateSurfaceData = function() {
        function get(name) {
            return parseFloat(document.getElementById(name).value);
        }

        const { vertices, uvs, indices } = generateSurface(get('A'), get('B'), get('N'), get('USeg'), get('VSeg'), get('VMin'), get('VMax'));
        const {normals, tangents} = calculateNormalsAndTangents(vertices, indices, uvs);
        this.BufferData(vertices, normals, tangents, uvs, indices);
    }
}
